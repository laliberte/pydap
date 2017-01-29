"""
Based on netcdf4-python

This code aims to provide:

A (partial) compatibility layer with netcdf4-python.
In order to do so code was directly
borrowed from the netcdf4-python package.

Frederic Laliberte, 2016
"""

import six
import numpy as np
import requests

from collections import OrderedDict

from ..exceptions import ServerError
from webob.exc import HTTPError
from ssl import SSLError
from ..client import open_url
from ..lib import DEFAULT_TIMEOUT
from ..cas import get_cookies

default_encoding = 'utf-8'

_private_atts = ['_grpid', '_grp', '_varid', 'groups', 'dimensions',
                 'variables', 'dtype', 'data_model', 'disk_format',
                 '_nunlimdim', 'path', 'parent', 'ndim', 'mask', 'scale',
                 'cmptypes', 'vltypes', 'enumtypes', '_isprimitive',
                 'file_format', '_isvlen', '_isenum', '_iscompound',
                 '_cmptype', '_vltype', '_enumtype', 'name',
                 '__orthogoral_indexing__', 'keepweakref', '_has_lsd',
                 '_url', '_timeout', '_session', '_application',
                 '_output_grid', '_verify', '_authentication_url',
                 '_username', '_password']


class Dataset(object):
    def __init__(self, url,
                 session=None, application=None,
                 timeout=DEFAULT_TIMEOUT,
                 verify=True,
                 authentication_uri=None,
                 username=None, password=None):
        """
        Mimics the netCDF4.Dataset API but is limited
        to OPeNDAP datasets.

        Args:
            url: string
                url to an OPeNDAP resource.
            session: requests.Session-like, optional
                a requests session object that may contain
                authentication cookies. In that case, these
                authentication cookies will be used to access
                restricted OPeNDAP resources.
            application: callable, optional
                a WSGI application. This application will be
                passed when calls are made to the OPeNDAP
                resource.
            timeout: float, optional
                maximum number of seconds before an OPeNDAP
                call raises a timeout error.
            verify: bool, optional
                flag to enable SSL verification in OPeNAP calls.
                Default: True
            authentication_uri: str or callable, optional
                if it is a callable, ``authentication_uri(url)``
                should return a string. That string or if
                ``authentication_uri`` is a string, it should be
                pointing to the central authentication service (CAS)
                for ``url``.
            username: str, optional
                if the CAS requires a user name, this string will be
                used.
            password: str, optional
                if the CAS requires a password, this string will be used.
        """
        self._url = url
        self._timeout = timeout
        self._session = session
        self._application = application

        # This API does not need Grid outputs:
        self._output_grid = False

        # Options for authentication:
        self._verify = verify
        self._authentication_uri = authentication_uri
        self._username = username
        self._password = password

        # Provided for compatibility:
        self.data_model = 'pyDAP'
        self.file_format = self.data_model
        self.disk_format = 'DAP2'
        self._isopen = 1
        self.path = '/'
        self.parent = None
        self.keepweakref = False

        # Assign dataset, allowing for
        # inadequate authentication:
        try:
            # First try without authentication
            self._assign_dataset()
        except HTTPError as e:
            if _maybe_auth_error(str(e)):
                # If first try and
                # 300 or 400 type error. Try to authenticate:
                try:
                    self._assign_dataset(authenticate=True)
                except HTTPError as e:
                    raise ServerError(str(e))
            else:
                raise ServerError(str(e))
        except (SSLError, requests.exceptions.SSLError):
            try:
                self._assign_dataset(authenticate=True)
            except (SSLError, requests.exceptions.SSLError):
                self._assign_dataset(authenticate=True, verify=False)

        self.dimensions = self._get_dims(self._pydap_dataset)
        self.variables = self._get_vars(self._pydap_dataset)

        self.groups = OrderedDict()

    def _assign_dataset(self, authenticate=False, verify=True):
        if authenticate:
            self._session = get_cookies.setup_session(self._authentication_uri,
                                                      username=self._username,
                                                      password=self._password,
                                                      session=self._session,
                                                      verify=min(self._verify,
                                                                 verify),
                                                      check_url=self._url)

        self._pydap_dataset = open_url(self._url, session=self._session,
                                       timeout=self._timeout,
                                       application=self._application,
                                       output_grid=self._output_grid)

    def __enter__(self):
        return self

    def __exit__(self, atype, value, traceback):
        self.close()

    def __getitem__(self, elem):
        # There are no groups. Simple mapping to variable:
        if elem in self.variables.keys():
            return self.variables[elem]
        else:
            raise IndexError('%s not found in %s' % (elem, self.path))

    def filepath(self):
        return self._url

    def __repr__(self):  # pragma: no cover
        if six.PY3:
            return self.__unicode__()
        else:
            return six.text_type(self).encode(default_encoding)

    def __unicode__(self):
        # taken directly from netcdf4-python netCDF4.pyx
        ncdump = ['%r\n' % type(self)]
        dimnames = tuple([_tostr(dimname)+'(%s)' %
                          len(self.dimensions[dimname])
                          for dimname in self.dimensions.keys()])
        varnames = tuple([_tostr(self.variables[varname].dtype) +
                          ' \033[4m' + _tostr(varname) + '\033[0m' +
                          (((_tostr(self.variables[varname].dimensions)
                             .replace("u'", ""))
                            .replace("'", ""))
                          .replace(", ", ","))
                          .replace(",)", ")")
                          for varname in sorted(self.variables.keys())])
        grpnames = tuple([_tostr(grpname)
                          for grpname in sorted(self.groups.keys())])
        if self.path == '/':
            ncdump.append('root group (%s data model, file format %s):\n' %
                          (self.data_model, self.disk_format))
        else:
            ncdump.append('group %s:\n' % self.path)
        attrs = ['    %s: %s\n' % (name, self.getncattr(name)) for name in
                 self.ncattrs()]
        ncdump = ncdump + attrs
        ncdump.append('    dimensions(sizes): %s\n' % ', '.join(dimnames))
        ncdump.append('    variables(dimensions): %s\n' % ', '.join(varnames))
        ncdump.append('    groups: %s\n' % ', '.join(grpnames))
        return ''.join(ncdump)

    def close(self):
        self._isopen = 0

    def isopen(self):
        return bool(self._isopen)

    def ncattrs(self):
        try:
            return self._pydap_dataset.attributes['NC_GLOBAL'].keys()
        except KeyError:
            return []

    def getncattr(self, attr):
        try:
            return self._pydap_dataset.attributes['NC_GLOBAL'][attr]
        except KeyError as e:
            raise AttributeError(str(e))

    def __getattr__(self, name):
        return self.getncattr(name)

    def set_auto_maskandscale(self, flag):
        raise NotImplementedError('set_auto_maskandscale is not '
                                  'implemented for pydap')

    def set_auto_mask(self, flag):
        raise NotImplementedError('set_auto_mask is not implemented for pydap')

    def set_auto_scale(self, flag):
        raise NotImplementedError('set_auto_scale is not implemented '
                                  'for pydap')

    def get_variables_by_attributes(self, **kwargs):
        # From netcdf4-python
        vs = []

        has_value_flag = False
        for vname in self.variables:
            var = self.variables[vname]
            for k, v in kwargs.items():
                if callable(v):
                    has_value_flag = v(getattr(var, k, None))
                    if not has_value_flag:
                        break
                # Must use getncattr
                elif hasattr(var, k) and var.getncattr(k) == v:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                vs.append(self.variables[vname])
        return vs

    def _get_dims(self, dataset):
        if ('DODS_EXTRA' in dataset.attributes.keys() and
           'Unlimited_Dimension' in dataset.attributes['DODS_EXTRA']):
            unlimited_dims = [dataset
                              .attributes['DODS_EXTRA']
                              ['Unlimited_Dimension']]
        else:
            unlimited_dims = []
        var_list = dataset.keys()
        var_id = np.argmax([len(dataset[varname].dimensions)
                            for varname in var_list])
        base_dimensions_list = dataset[var_list[var_id]].dimensions
        try:
            base_dimensions_lengths = dataset[var_list[var_id]].array.shape
        except (AttributeError, KeyError):
            # KeyError is important for python 3.3 and 3.4
            base_dimensions_lengths = dataset[var_list[var_id]].shape

        for varname in var_list:
            if not (set(base_dimensions_list)
                    .issuperset(dataset[varname].dimensions)):
                for dim_id, dim in enumerate(dataset[varname].dimensions):
                    if dim not in base_dimensions_list:
                        base_dimensions_list += (dim,)
                        base_dimensions_lengths += (dataset[varname]
                                                    .shape[dim_id],)
        dimensions_dict = OrderedDict()
        for dim, dim_length in zip(base_dimensions_list,
                                   base_dimensions_lengths):
            dimensions_dict[dim] = Dimension(self, dim,
                                             size=dim_length,
                                             isunlimited=(dim
                                                          in unlimited_dims))
        return dimensions_dict

    def _get_vars(self, dataset):
        return dict([(var, Variable(dataset[var], var, self))
                     for var in dataset.keys()])


class Variable(object):
    def __init__(self, var, name, grp):
        self._grp = grp
        self._var = var
        self.dimensions = self._getdims()

        self.datatype = self.dtype
        self.ndim = len(self.dimensions)
        self.scale = True
        self.name = name
        self.size = np.prod(self.shape)

    @property
    def shape(self):
        return self._get_array_att('shape')

    @property
    def dtype(self):
        if self._get_array_att('dtype').char == 'S':
            if ('DODS' in self.ncattrs() and
                'dimName' in self.getncattr('DODS') and
               self.getncattr('DODS')['dimName'] in self.getncattr('DODS')):
                return np.dtype('S' +
                                str(self.getncattr('DODS')
                                    [self.getncattr('DODS')['dimName']]))
            else:
                # Default to length 100 strings:
                return np.dtype('S100')
        else:
            return np.dtype(self._get_array_att('dtype'))

    def chunking(self):
        return 'contiguous'

    def filters(self):
        return None

    def get_var_chunk_cache(self):
        raise NotImplementedError('get_var_chunk_cache is not implemented')

    def ncattrs(self):
        return self._var.attributes.keys()

    def getncattr(self, attr):
        try:
            return self._var.attributes[attr]
        except KeyError as e:
            raise AttributeError(str(e))

    def __getattr__(self, name):
        return self.getncattr(name)

    def getValue(self):
        return self._var[...]

    def group(self):
        return self._grp

    def __array__(self):
        return self[...]

    def __repr__(self):  # pragma: no cover
        if six.PY3:
            return self.__unicode__()
        else:
            return six.text_type(self).encode(default_encoding)

    def __getitem__(self, key):
        try:
            return _getitem_safe(self, key)
        except HTTPError as e:
            # Before raising error, try to authenticate:
            if _maybe_auth_error(str(e)):
                try:
                    self._grp._assign_dataset(authenticate=True)
                    return _getitem_safe(self, key)
                except HTTPError as e:
                    raise ServerError(str(e))
            else:
                raise ServerError(str(e))
        except (SSLError, requests.exceptions.SSLError):
            try:
                self._grp._assign_dataset(authenticate=True)
            except (SSLError, requests.exceptions.SSLError):
                self._grp._assign_dataset(authenticate=True, verify=False)
            try:
                return _getitem_safe(self, key)
            except HTTPError as e:
                raise ServerError(str(e))

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized object')
        else:
            return self.shape[0]

    def set_auto_maskandscale(self, maskandscale):
        raise NotImplementedError('set_auto_maskandscale is '
                                  'not implemented for pydap')

    def set_auto_scale(self, scale):
        raise NotImplementedError('set_auto_scale is not implemented '
                                  'for pydap')

    def set_auto_mask(self, mask):
        raise NotImplementedError('set_auto_mask is not implemented for pydap')

    def __unicode__(self):
        # taken directly from netcdf4-python: netCDF4.pyx
        ncdump_var = ['%r\n' % type(self)]
        dimnames = tuple([_tostr(dimname)
                          for dimname in self.dimensions])
        attrs = ['    %s: %s\n' % (name, self.getncattr(name)) for name in
                 self.ncattrs()]
        ncdump_var.append('%s %s(%s)\n' %
                          (self.dtype, self.name, ', '.join(dimnames)))
        ncdump_var = ncdump_var + attrs
        unlimdims = []
        for dimname in self.dimensions:
            dim = self._grp.dimensions[dimname]
            if dim.isunlimited():
                unlimdims.append(dimname)
        ncdump_var.append('unlimited dimensions: %s\n' % ', '.join(unlimdims))
        ncdump_var.append('current shape = %s\n' % repr(self.shape))
        return ''.join(ncdump_var)

    def _getdims(self):
        return self._var.dimensions

    def _get_array_att(self, att):
        if hasattr(self._var, att):
            return getattr(self._var, att)
        else:
            return getattr(self._var.array, att)


class Dimension(object):
    def __init__(self, grp, name, size=0, isunlimited=True):
        self._grp = grp

        self.size = size
        self._isunlimited = isunlimited

        self._name = name

    def __len__(self):
        return self.size

    def isunlimited(self):
        return self._isunlimited

    def group(self):
        return self._grp

    def __repr__(self):  # pragma: no cover
        if six.PY3:
            return self.__unicode__()
        else:
            return six.text_type(self).encode(default_encoding)

    def __unicode__(self):
        # taken directly from netcdf4-python: netCDF4.pyx
        if self.isunlimited():
            return (repr(type(self)) +
                    " (unlimited): name = '%s', size = %s\n" %
                    (self._name, len(self)))
        else:
            return (repr(type(self))+": name = '%s', size = %s\n" %
                    (self._name, len(self)))


def _tostr(s):
    try:
        ss = str(s)
    except Exception:  # pragma: no cover
        ss = s
    return ss


def _maybe_auth_error(message):
    if len(message) < 3:
        return False
    try:
        code = int(message[:3])
    except TypeError:
        return False
    if 300 <= code < 500:
        return True
    else:
        return False


def _getitem_safe(dataset, key):
    try:
        return dataset._var.array.__getitem__(key)
    except (AttributeError, KeyError,
            ServerError, HTTPError):
        # KeyError is important for python 3.3 and 3.4
        if (isinstance(key, slice) and
           key == slice(None, None, None)):
            # A single dimension ellipsis was requested.
            # Use netCDF4 convention:
            return dataset[...]
        else:
            return dataset._var.__getitem__(key)
