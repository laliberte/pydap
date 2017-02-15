"""Test the DAP handler, which forms the core of the client."""

import netCDF4
import tempfile
import os
import numpy as np
from six.moves import zip

from pydap.handlers.netcdf import NetCDFHandler
from pydap.apis.netCDF4 import Dataset
from pydap.wsgi.ssf import ServerSideFunctions
from pydap.cas import esgf
from pydap.net import follow_redirect

import unittest
from nose.plugins.attrib import attr


class MockErrors:
    def __init__(self, errors):
        self.errors = errors
        self.error_id = 0

    def __call__(self, *args, **kwargs):
        self.error_id += 1
        raise self.errors[min(self.error_id - 1,
                              len(self.errors) - 1)]


def _message(e):
    try:
        return e.exception.message
    except AttributeError:
        return str(e.exception)


def _compare_repr(a, b):  # pragma: no cover
    import difflib
    print('{} => {}'.format(a,b))  
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0]==' ': continue
        elif s[0]=='-':
            print(u'Delete "{}" from position {}'.format(s[-1],i))
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i))    
    print()    


class TestDataset(unittest.TestCase):

    """Test that the handler creates the correct dataset from a URL."""
    data = [(10, 15.2, 'Diamond_St'),
            (11, 13.1, 'Blacktail_Loop'),
            (12, 13.3, 'Platinum_St'),
            (13, 12.1, 'Kodiak_Trail')]

    def setUp(self):
        """Create WSGI apps"""

        # Create tempfile:
        fileno, self.test_file = tempfile.mkstemp(suffix='.nc')
        # must close file number:
        os.close(fileno)
        with netCDF4.Dataset(self.test_file, 'w') as output:
            output.createDimension('index', None)
            for dim in ['lat', 'lon']:
                output.createDimension(dim, 1)
                output.createVariable(dim, 'f', (dim,))
            dim_tuple = ('index', 'lat', 'lon')
            temp = output.createVariable('index', '<i4', dim_tuple)
            split_data = zip(*self.data)
            temp[:] = next(split_data)
            temp = output.createVariable('temperature', '<f8', dim_tuple)
            temp[:] = next(split_data)
            temp = output.createVariable('station', 'S40', dim_tuple)
            temp.setncattr('long_name', 'Station Name')
            for item_id, item in enumerate(next(split_data)):
                temp[item_id, 0, 0] = item
            output.createDimension('bnds', 1)
            temp = output.createVariable('bnds', '<i4', ('bnds',))
            output.createVariable('index_bnds', '<f8', ('index', 'bnds'))
            output.setncattr('history', 'test file for netCDF4 api')
        self.app = ServerSideFunctions(NetCDFHandler(self.test_file))

    def test_dataset_direct(self):
        """Test that dataset has the correct data proxies for grids."""
        dtype = [('index', '<i4'),
                 ('temperature', '<f8'),
                 ('station', 'S40')]
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            retrieved_data = list(zip(np.squeeze(dataset['index'][:]),
                                      np.squeeze(dataset['temperature'][:]),
                                      np.squeeze(dataset['station'][:])))
        np.testing.assert_array_equal(np.array(retrieved_data, dtype=dtype),
                                      np.array(self.data, dtype=dtype))

    def test_dataset_missing_elem(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(IndexError) as e:
                dataset['missing']
            assert str(_message(e)) == 'missing not found in /'

    def test_dataset_httperror(self):
        from webob.exc import HTTPError
        from pydap.exceptions import ServerError

        mock_httperror = MockErrors([HTTPError('400 Test Error')])

        with self.assertRaises(ServerError) as e:
            Dataset('http://localhost:8000/',
                    application=mock_httperror)
        assert str(e.exception.value) == '400 Test Error'

        mock_httperror = MockErrors([HTTPError('500 Test Error')])

        with self.assertRaises(ServerError) as e:
            Dataset('http://localhost:8000/',
                    application=mock_httperror)
        assert str(e.exception.value) == '500 Test Error'

    def test_dataset_sslerror(self):
        from ssl import SSLError

        mock_sslerror = MockErrors([SSLError('SSL Test Error')])

        with self.assertRaises(SSLError) as e:
            Dataset('http://localhost:8000/',
                    application=mock_sslerror)
        assert str(e.exception) == "('SSL Test Error',)"

    def test_variable_httperror(self):
        from webob.exc import HTTPError
        from pydap.exceptions import ServerError

        mock_httperror = MockErrors([HTTPError('400 Test Error'),
                                     HTTPError('401 Test Error')])

        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            variable._var.array.__getitem__ = mock_httperror
            variable._var.__getitem__ = mock_httperror
            with self.assertRaises(ServerError) as e:
                variable[...]
            assert str(e.exception.value) == '401 Test Error'

        mock_httperror = MockErrors([HTTPError('400 Test Error'),
                                     HTTPError('500 Test Error')])

        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            variable._var.array.__getitem__ = mock_httperror
            variable._var.__getitem__ = mock_httperror
            with self.assertRaises(ServerError) as e:
                variable[...]
            assert str(e.exception.value) == '500 Test Error'

    def test_variable_sslerror(self):
        from webob.exc import HTTPError
        from ssl import SSLError
        from pydap.exceptions import ServerError

        mock_sslerror = MockErrors([HTTPError('400 Test Error'),
                                    SSLError('SSL Test Error'),
                                    HTTPError('500 Test Error')])

        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            variable._var.array.__getitem__ = mock_sslerror
            variable._var.__getitem__ = mock_sslerror
            with self.assertRaises(ServerError) as e:
                variable[...]
            assert str(e.exception.value) == '500 Test Error'

        mock_sslerror = MockErrors([HTTPError('400 Test Error'),
                                    SSLError('SSL Test Error'),
                                    HTTPError('500 Test Error')])
        mock_assignerror = MockErrors([SSLError('SSL dataset Error')])

        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            dataset._assign_dataset = mock_assignerror
            variable = dataset.variables['temperature']
            variable._var.array.__getitem__ = mock_sslerror
            variable._var.__getitem__ = mock_sslerror
            with self.assertRaises(SSLError) as e:
                variable[...]
            assert str(e.exception) == "('SSL dataset Error',)"

    def test_dataset_filepath(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.filepath() == 'http://localhost:8000/'

    def test_dataset_repr(self):
        expected_repr = """<class 'pydap.apis.netCDF4.Dataset'>
root group (pyDAP data model, file format DAP2):
    history: test file for netCDF4 api
    dimensions(sizes): index(4), lat(1), lon(1), bnds(1)
    variables(dimensions): >i4 \033[4mbnds\033[0m(bnds), >i4 \033[4mindex\033[0m(), >f8 \033[4mindex_bnds\033[0m(index,bnds), >f4 \033[4mlat\033[0m(lat), >f4 \033[4mlon\033[0m(lon), |S100 \033[4mstation\033[0m(index,lat,lon), >f8 \033[4mtemperature\033[0m(index,lat,lon)
    groups: 
"""
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            # _compare_repr(repr(dataset), expected_repr)
            assert repr(dataset) == expected_repr
            dataset.path = 'test/'
            expected_repr = '\n'.join(
                                  [line if line_id != 1 else 'group test/:'
                                   for line_id, line
                                   in enumerate(expected_repr.split('\n'))])
            assert repr(dataset) == expected_repr

    def test_dataset_isopen(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.isopen()

    def test_dataset_ncattrs(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert list(dataset.ncattrs()) == ['history']
            del dataset._pydap_dataset.attributes['NC_GLOBAL']
            assert list(dataset.ncattrs()) == []

    def test_dataset_getattr(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.getncattr('history') == 'test file for netCDF4 api'
            assert getattr(dataset, 'history') == 'test file for netCDF4 api'
            with self.assertRaises(AttributeError) as e:
                getattr(dataset, 'inexistent')
            assert str(_message(e)) == "'inexistent'"

    def test_dataset_set_auto_maskandscale(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                dataset.set_auto_maskandscale(True)
            assert str(_message(e)) == ('set_auto_maskandscale is not '
                                        'implemented for pydap')

    def test_dataset_set_auto_mask(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                dataset.set_auto_mask(True)
            assert str(_message(e)) == ('set_auto_mask is not '
                                        'implemented for pydap')

    def test_dataset_set_auto_scale(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                dataset.set_auto_scale(True)
            assert str(_message(e)) == ('set_auto_scale is not '
                                        'implemented for pydap')

    def test_dataset_get_variable_by_attribute(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            var = dataset.get_variables_by_attributes(**{'long_name':
                                                         'Station Name'})
            assert var == [dataset.variables['station']]
            
            def station(x):
                try:
                    return 'Station' in x
                except TypeError:
                    return False
            var = dataset.get_variables_by_attributes(**{'long_name':
                                                         station})
            assert var == [dataset.variables['station']]

            def inexistent(x):
                return False

            assert callable(inexistent)
            var = dataset.get_variables_by_attributes(**{'long_name':
                                                         inexistent})
            assert var == []

    def test_dimension_unlimited(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.dimensions['index'].isunlimited()

    def test_dimension_group(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.dimensions['index'].group() == dataset

    def test_dimension_unlimited_repr(self):
        expected_repr = ("<class 'pydap.apis.netCDF4.Dimension'> (unlimited): "
                         "name = 'index', size = 4")
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.dimensions['index'].isunlimited()
            assert repr(dataset
                        .dimensions['index']).strip() == expected_repr

    def test_variable_group(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            assert variable.group() == dataset

    def test_variable_length(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            assert len(variable) == 4

            def mock_shape(x):
                return None

            variable._get_array_att = mock_shape
            with self.assertRaises(TypeError) as e:
                len(variable)
            assert str(_message(e)) == 'len() of unsized object'

    def test_variable_repr(self):
        expected_repr = """<class 'pydap.apis.netCDF4.Variable'>
|S100 station(index, lat, lon)
    long_name: Station Name
unlimited dimensions: index
current shape = (4, 1, 1)
"""
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['station']
            #_compare_repr(expected_repr, repr(variable))
            assert repr(variable) == expected_repr

            # Mock unlimited dimension:
            assert dataset.dimensions['index'].isunlimited()
            assert repr(variable) == expected_repr

    def test_variable_hdf5_properties(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            assert variable.chunking() == 'contiguous'
            assert variable.filters() is None
            with self.assertRaises(NotImplementedError) as e:
                variable.get_var_chunk_cache()
            assert str(_message(e)) == ('get_var_chunk_cache is not '
                                                'implemented')

    def test_variable_set_auto_maskandscale(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                variable = dataset.variables['temperature']
                variable.set_auto_maskandscale(True)
            assert str(_message(e)) == ('set_auto_maskandscale is not '
                                                'implemented for pydap')

    def test_variable_set_auto_mask(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                variable = dataset.variables['temperature']
                variable.set_auto_mask(True)
            assert str(_message(e)) == ('set_auto_mask is not '
                                                'implemented for pydap')

    def test_variable_set_auto_scale(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            with self.assertRaises(NotImplementedError) as e:
                variable = dataset.variables['temperature']
                variable.set_auto_scale(True)
            assert str(_message(e)) == ('set_auto_scale is not '
                                                'implemented for pydap')

    def test_variable_get(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['temperature']
            assert np.all(variable[:] == variable[...])
            assert np.all(variable[:] == np.asarray(variable))
            assert np.all(variable[:] == variable.getValue())

    def test_variable_string_dtype(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['station']
            assert variable.dtype != 'S40'
            assert 'DODS' not in variable._var.attributes
            variable._var.attributes['DODS'] = {'dimName': 'string',
                                                'string': 40}
            assert variable.dtype == 'S40'

    def tearDown(self):
        os.remove(self.test_file)


@attr('auth')
@attr('prod_url')
class TestESGFDataset(unittest.TestCase):
    url = ('http://aims3.llnl.gov/thredds/dodsC/'
           'cmip5_css02_data/cmip5/output1/CCCma/CanCM4/'
           'decadal1995/fx/atmos/fx/r0i0p0/orog/1/'
           'orog_fx_CanCM4_decadal1995_r0i0p0.nc')
    test_url = url + '.dods?orog[0:1:4][0:1:4]'


    def test_variable_esgf_session(self):
        """
        This test makes sure that passing a authenticated ESGF session
        to Dataset will allow the retrieval of data.
        """
        assert(os.environ.get('OPENID_ESGF'))
        assert(os.environ.get('PASSWORD_ESGF'))
        session = esgf.setup_session(os.environ.get('OPENID_ESGF'),
                                     os.environ.get('PASSWORD_ESGF'),
                                     check_url=self.url)
        # Ensure authentication:
        res = follow_redirect(self.test_url, session=session)
        assert(res.status_code == 200)

        # This server does not access retrieval of grid coordinates.
        # The option output_grid disables this, reverting to
        # pydap 3.1.1 behavior. For older OPeNDAP servers (i.e. ESGF),
        # this appears necessary.
        dataset = Dataset(self.url, session=session)
        data = dataset['orog'][50:55, 50:55]
        expected_data = [[197.70425, 16.319595, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [677.014, 628.29675, 551.06, 455.5758, 343.7354],
                         [1268.3304, 1287.9553, 1161.0402, 978.3153, 809.143]]
        assert(np.isclose(data, expected_data).all())

    def test_variable_esgf_auth(self):
        """
        This test makes sure that passing an EGSF password
        and an ESGF authentication link to Dataset will allow
        the retrieval of data.
        """
        assert(os.environ.get('OPENID_ESGF'))
        assert(os.environ.get('PASSWORD_ESGF'))
        session = esgf.setup_session(os.environ.get('OPENID_ESGF'),
                                     os.environ.get('PASSWORD_ESGF'),
                                     check_url=self.url)
        # Ensure authentication:
        res = follow_redirect(self.test_url, session=session)
        assert(res.status_code == 200)

        # This server does not access retrieval of grid coordinates.
        # The option output_grid disables this, reverting to
        # pydap 3.1.1 behavior. For older OPeNDAP servers (i.e. ESGF),
        # this appears necessary.
        dataset = Dataset(self.url, password=os.environ.get('PASSWORD_ESGF'),
                          authentication_uri=esgf._uri(os.environ.get('OPENID_ESGF')))
        data = dataset['orog'][50:55, 50:55]
        expected_data = [[197.70425, 16.319595, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [677.014, 628.29675, 551.06, 455.5758, 343.7354],
                         [1268.3304, 1287.9553, 1161.0402, 978.3153, 809.143]]
        assert(np.isclose(data, expected_data).all())
