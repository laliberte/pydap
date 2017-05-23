"""A handler for remote datasets.

DAP handlers convert from different data formats (NetCDF, eg) to the internal
Pydap model. The Pydap client is just a handler that converts from a remote
dataset to the internal model.

"""

import io
import sys
import pprint
import copy

# handlers should be set by the application
# http://docs.python.org/2/howto/logging.html#configuring-logging-for-a-library
import logging
from six.moves.urllib.parse import urlsplit, urlunsplit

from ...model import (BaseType,
                      SequenceType,
                      GridType)
from ...net import GET, raise_for_status
from ...lib import fix_slice, walk
from ..lib import BaseHandler
from ...parsers.dds import build_dataset
from ...parsers.das import parse_das, add_attributes
from ...parsers import parse_ce
from model import unpack_data, SequenceProxy, BaseProxy

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class DAPHandler(BaseHandler):

    """Build a dataset from a DAP base URL."""

    def __init__(self, url, application=None, session=None, output_grid=True):
        # download DDS/DAS
        scheme, netloc, path, query, fragment = urlsplit(url)

        ddsurl = urlunsplit((scheme, netloc, path + '.dds', query, fragment))
        r = GET(ddsurl, application, session)
        raise_for_status(r)
        if not r.charset:
            r.charset = 'ascii'
        dds = r.text

        dasurl = urlunsplit((scheme, netloc, path + '.das', query, fragment))
        r = GET(dasurl, application, session)
        raise_for_status(r)
        if not r.charset:
            r.charset = 'ascii'
        das = r.text

        # build the dataset from the DDS and add attributes from the DAS
        self.dataset = build_dataset(dds)
        add_attributes(self.dataset, parse_das(das))

        # remove any projection from the url, leaving selections
        projection, selection = parse_ce(query)
        url = urlunsplit((scheme, netloc, path, '&'.join(selection), fragment))

        # now add data proxies
        for var in walk(self.dataset, BaseType):
            var.data = BaseProxy(url, var.id, var.dtype, var.shape,
                                 application=application,
                                 session=session)
        for var in walk(self.dataset, SequenceType):
            template = copy.copy(var)
            var.data = SequenceProxy(url, template, application=application,
                                     session=session)

        # apply projections
        for var in projection:
            target = self.dataset
            while var:
                token, index = var.pop(0)
                target = target[token]
                if isinstance(target, BaseType):
                    target.data.slice = fix_slice(index, target.shape)
                elif isinstance(target, GridType):
                    index = fix_slice(index, target.array.shape)
                    target.array.data.slice = index
                    for s, child in zip(index, target.maps):
                        target[child].data.slice = (s,)
                elif isinstance(target, SequenceType):
                    target.data.slice = index

        # retrieve only main variable for grid types:
        for var in walk(self.dataset, GridType):
            var.set_output_grid(output_grid)


def dump():  # pragma: no cover
    """Unpack dods response into lists.

    Return pretty-printed data.

    """
    dods = sys.stdin.read()
    dds, xdrdata = dods.split(b'\nData:\n', 1)
    xdr_stream = io.BytesIO(xdrdata)
    dds = dds.decode('ascii')
    dataset = build_dataset(dds)
    dataset.data = unpack_data(xdr_stream, dataset)

    pprint.pprint(dataset.data)
