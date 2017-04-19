"""A handler for remote datasets.

DAP handlers convert from different data formats (NetCDF, eg) to the internal
Pydap model. The Pydap client is just a handler that converts from a remote
dataset to the internal model.

"""

import copy
import re
from itertools import chain

# handlers should be set by the application
# http://docs.python.org/2/howto/logging.html#configuring-logging-for-a-library
import logging
import numpy as np
from six.moves.urllib.parse import urlsplit, urlunsplit, quote
from six import text_type

from ...model import (BaseType,
                      SequenceType, StructureType)
from ...net import GET, raise_for_status
from ...lib import (
    encode, combine_slices, fix_slice, hyperslab,
    START_OF_SEQUENCE, StreamReader, BytesReader)
from ..lib import ConstraintExpression, IterData
from ...parsers.dds import build_dataset

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class DapProxy(object):

    """
    A proxy for remote dap types.
    """

    def __init__(self, baseurl, template, application=None,
                 session=None):
        self.baseurl = baseurl
        self.template = template
        self.application = application
        self.session = session

    def __repr__(self):
        return 'DapProxy(%s)' % ', '.join(
            map(repr, [
                self.baseurl, self.template]))


class DatasetProxy(DapProxy):
    """ A proxy for remote Dataset types """
    def __init__(self, baseurl, template, selection=None,
                 application=None, session=None):
        super(DatasetProxy, self).__init__(baseurl, template, selection,
                                           application, session)
        self.selection = selection or []

    def __repr__(self):
        return 'DatasetProxy(%s)' % ', '.join(
            map(repr, [self.baseurl, self.template]))

    def __copy__(self):
        """Return a lightweight copy of the object."""
        return type(self)(self.baseurl, self.template, self.selection[:],
                          self.application, self.session)

    @property
    def dtype(self):
        return self.template.dtype

    @property
    def shape(self):
        return self.template.shape

    def __getitem__(self, index):
        # build download url
        out = copy.copy(self)
        out.slice = combine_slices(self.slice, fix_slice(index, out.shape))
        return out.data

    def get_url(self):
        # build download url
        url = self.url
        _logger.info("Fetching URL: %s" % url)

        # download and unpack data
        r = GET(url, self.application, self.session)
        raise_for_status(r)
        return r

    @property
    def data(self):
        r = self.get_url()
        dds, data = r.body.split(b'\nData:\n', 1)
        dds = dds.decode(r.content_encoding or 'ascii')

        # Parse received dataset:
        dataset = build_dataset(dds)
        return unpack_data(BytesReader(data), dataset)

    def __iter__(self):
        return self.iterdata()

    def iterdata(self):
        r = self.get_url()

        i = r.app_iter
        if not hasattr(i, '__next__'):
            i = iter(i)

        # Fast forward past the DDS header, but keep it around:
        # the pattern could span chunk boundaries though so make sure to check
        pattern = b'Data:\n'
        dds, last_chunk = find_pattern_in_string_iter(pattern, i)

        if last_chunk is None:
            raise ValueError("Could not find data segment in response from {}"
                             .format(self.url))

        dds = dds.decode(r.content_encoding or 'ascii')
        # Parse received DDS:
        dataset = build_dataset(dds)

        # Then construct a stream consisting of everything from
        # 'Data:\n' to the end of the chunk + the rest of the stream
        def stream_start():
            yield last_chunk

        stream = StreamReader(chain(stream_start(), i))
        return unpack_sequence(stream, dataset)

    @property
    def url(self):
        """Return url from where data is fetched."""
        scheme, netloc, path, query, fragment = urlsplit(self.baseurl)
        url = urlunsplit((
            scheme, netloc, path + '.dods',
            self.id + hyperslab(self.slice) + '&' + query + '&' +
            '&'.join(self.selection), fragment)).rstrip('&')
        return url

    @property
    def id(self):
        """Return the id of this sequence."""
        if list(self._all_keys()) == list(self.keys()):
            return quote(self.template.id)
        else:
            return ','.join(quote(child.id) for child
                            in self.template.children())


class StructureProxy(DatasetProxy):
    """ A proxy for remote Dataset types """
    def __init__(self, baseurl, template, selection=None, slice_=None,
                 application=None, session=None):
        super(StructureProxy, self).__init__(baseurl, template, selection,
                                             application, session)
        self.slice = slice_ or tuple(slice(None) for s in self.shape)

    def __repr__(self):
        return 'StructureProxy(%s)' % ', '.join(
            map(repr, [self.baseurl, self.template, self.slice]))

    def __copy__(self):
        """Return a lightweight copy of the object."""
        return type(self)(self.baseurl, self.template, self.selection[:],
                          self.slice, self.application, self.session)


class GridProxy(StructureProxy):
    """A proxy for remote base types.

    This class behaves like a Numpy array, proxying the data from a base type
    on a remote dataset.

    """

    def __init__(self, baseurl, template, selection=None, slice_=None,
                 application=None, session=None):
        super(GridProxy, self).__init__(baseurl, template, selection, slice_,
                                        application, session)

    def __repr__(self):
        return 'GridProxy(%s)' % ', '.join(
            map(repr, [self.baseurl, self.template, self.slice]))

    def __len__(self):
        return int(np.prod(self.shape))


class BaseProxy(GridProxy):
    """A proxy for remote base types.

    This class behaves like a Numpy array, proxying the data from a base type
    on a remote dataset.

    """

    def __init__(self, baseurl, template, slice_=None,
                 application=None, session=None):
        super(BaseProxy, self).__init__(baseurl, template, slice_,
                                        application=application,
                                        session=session)

    def __repr__(self):
        return 'BaseProxy(%s)' % ', '.join(
            map(repr, [self.baseurl, self.template, self.slice]))

    # Comparisons return a boolean array
    def __eq__(self, other):
        return self[:] == other

    def __ne__(self, other):
        return self[:] != other

    def __ge__(self, other):
        return self[:] >= other

    def __le__(self, other):
        return self[:] <= other

    def __gt__(self, other):
        return self[:] > other

    def __lt__(self, other):
        return self[:] < other


class SequenceProxy(StructureProxy):

    """A proxy for remote sequences.

    This class behaves like a Numpy structured array, proxying the data from a
    sequence on a remote dataset. The data is streamed from the dataset,
    meaning it can be treated one record at a time before the whole data is
    downloaded.

    """

    def __init__(self, baseurl, template, selection=None, slice_=None,
                 application=None, session=None):
        super(BaseProxy, self).__init__(baseurl, template, selection,
                                        slice_, application, session)

    def __repr__(self):
        return 'SequenceProxy(%s)' % ', '.join(
            map(repr, [self.baseurl, self.template, self.selection,
                       self.slice]))

    def __getitem__(self, key):
        """Return a new object representing a subset of the data."""
        out = copy.copy(self)

        try:
            out.template = out.template[key]
        except KeyError:
            # return a copy with the added constraints
            if isinstance(key, ConstraintExpression):
                out.selection.extend(str(key).split('&'))

            # slice data
            else:
                if isinstance(key, int):
                    key = slice(key, key+1)
                out.slice = combine_slices(self.slice, (key,))
        return out

    def __eq__(self, other):
        return ConstraintExpression('%s=%s' % (self.id, encode(other)))

    def __ne__(self, other):
        return ConstraintExpression('%s!=%s' % (self.id, encode(other)))

    def __ge__(self, other):
        return ConstraintExpression('%s>=%s' % (self.id, encode(other)))

    def __le__(self, other):
        return ConstraintExpression('%s<=%s' % (self.id, encode(other)))

    def __gt__(self, other):
        return ConstraintExpression('%s>%s' % (self.id, encode(other)))

    def __lt__(self, other):
        return ConstraintExpression('%s<%s' % (self.id, encode(other)))


def unpack_sequence(stream, template):
    """Unpack data from a sequence, yielding records."""
    # is this a sequence or a base type?
    sequence = isinstance(template, SequenceType)

    # if there are no children, we use the template as the only column
    cols = list(template.children()) or [template]

    # if there are no strings and no nested sequences we can unpack record by
    # record easily
    simple = all(isinstance(c, BaseType) and c.dtype.char not in "SU"
                 for c in cols)

    if simple:
        dtype = np.dtype([("", c.dtype, c.shape) for c in cols])
        marker = stream.read(4)
        while marker == START_OF_SEQUENCE:
            rec = np.fromstring(stream.read(dtype.itemsize), dtype=dtype)[0]
            if not sequence:
                rec = rec[0]
            yield rec
            marker = stream.read(4)
    else:
        marker = stream.read(4)
        while marker == START_OF_SEQUENCE:
            rec = unpack_children(stream, template)
            if not sequence:
                rec = rec[0]
            else:
                rec = tuple(rec)
            yield rec
            marker = stream.read(4)


def unpack_children(stream, template):
    """Unpack children from a structure, returning their data."""
    cols = list(template.children()) or [template]

    out = []
    for col in cols:
        # sequences and other structures
        if isinstance(col, SequenceType):
            out.append(IterData(list(unpack_sequence(stream, col)), col))
        elif isinstance(col, StructureType):
            out.append(tuple(unpack_children(stream, col)))

        # unpack arrays
        else:
            out.extend(convert_stream_to_list(stream, col.dtype, col.shape,
                                              col.id))
    return out


def convert_stream_to_list(stream, dtype, shape, id):
    out = []
    if shape:
        n = np.fromstring(stream.read(4), ">I")[0]
        count = dtype.itemsize * n
        if dtype.char in "SU":
            data = []
            for _ in range(n):
                k = np.fromstring(stream.read(4), ">I")[0]
                data.append(stream.read(k))
                stream.read(-k % 4)
            out.append(np.array([text_type(x.decode('ascii'))
                                 for x in data], 'S').reshape(shape))
        else:
            stream.read(4)  # read additional length
            try:
                out.append(
                    np.fromstring(
                        stream.read(count), dtype).reshape(shape))
            except ValueError as e:
                if str(e) == 'total size of new array must be unchanged':
                    # server-side failure.
                    # it is expected that the user should be mindful of this:
                    raise RuntimeError(
                                ('variable {0} could not be properly '
                                 'retrieved. To avoid this '
                                 'error consider using open_url(..., '
                                 'output_grid=False).').format(quote(id)))
                else:
                    raise
            if dtype.char == "B":
                stream.read(-n % 4)

    # special types: strings and bytes
    elif dtype.char in 'SU':
        k = np.fromstring(stream.read(4), '>I')[0]
        out.append(text_type(stream.read(k).decode('ascii')))
        stream.read(-k % 4)
    elif dtype.char == 'B':
        data = np.fromstring(stream.read(1), dtype)[0]
        stream.read(3)
        out.append(data)
    # usual data
    else:
        out.append(
            np.fromstring(stream.read(dtype.itemsize), dtype)[0])
    return out


def unpack_data(xdr_stream, dataset):
    """Unpack a string of encoded data, returning data as lists."""
    return unpack_children(xdr_stream, dataset)


def find_pattern_in_string_iter(pattern, i):
    dds = b''
    last_chunk = b''
    length = len(pattern)
    for this_chunk in i:
        last_chunk += this_chunk
        m = re.search(pattern, last_chunk)
        if m:
            dds += last_chunk[:m.end()]
            return dds, last_chunk[m.end():]
        dds += this_chunk
        last_chunk = last_chunk[-length:]
