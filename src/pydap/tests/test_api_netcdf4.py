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
            temp = output.createVariable('index', '<i4', ('index',))
            split_data = zip(*self.data)
            temp[:] = next(split_data)
            temp = output.createVariable('temperature', '<f8', ('index',))
            temp[:] = next(split_data)
            temp = output.createVariable('station', 'S40', ('index',))
            temp.setncattr('long_name', 'Station Name')
            for item_id, item in enumerate(next(split_data)):
                temp[item_id] = item
            output.createDimension('tag', 1)
            temp = output.createVariable('tag', '<i4', ('tag',))
            output.setncattr('history', 'test file for netCDF4 api')
        self.app = ServerSideFunctions(NetCDFHandler(self.test_file))

    def test_dataset_direct(self):
        """Test that dataset has the correct data proxies for grids."""
        dtype = [('index', '<i4'),
                 ('temperature', '<f8'),
                 ('station', 'S40')]
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            retrieved_data = list(zip(dataset['index'][:],
                                      dataset['temperature'][:],
                                      dataset['station'][:]))
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
    dimensions(sizes): index(4), tag(1)
    variables(dimensions): >i4 \033[4mindex\033[0m(index), |S100 \033[4mstation\033[0m(index), >i4 \033[4mtag\033[0m(tag), >f8 \033[4mtemperature\033[0m(index)
    groups: 
"""
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
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
            assert not dataset.dimensions['index'].isunlimited()
            assert isinstance(dataset._pydap_dataset
                              .attributes['DODS_EXTRA'], dict)
            assert 'Unlimited_Dimension' not in (dataset
                                                 ._pydap_dataset
                                                 .attributes['DODS_EXTRA'])
            (dataset._pydap_dataset
             .attributes['DODS_EXTRA']
             .update({'Unlimited_Dimension': 'index'}))
            dataset.dimensions = dataset._get_dims(dataset._pydap_dataset)
            assert dataset.dimensions['index'].isunlimited()

    def test_dimension_group(self):
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert dataset.dimensions['index'].group() == dataset

    def test_dimension_repr(self):
        expected_repr = ("<class 'pydap.apis.netCDF4.Dimension'>: "
                         "name = 'index', size = 4")
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert repr(dataset
                        .dimensions['index']).strip() == expected_repr

    def test_dimension_unlimited_repr(self):
        expected_repr = ("<class 'pydap.apis.netCDF4.Dimension'> (unlimited): "
                         "name = 'index', size = 4")
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            assert not dataset.dimensions['index'].isunlimited()
            assert isinstance(dataset._pydap_dataset
                              .attributes['DODS_EXTRA'], dict)
            assert 'Unlimited_Dimension' not in (dataset
                                                 ._pydap_dataset
                                                 .attributes['DODS_EXTRA'])
            (dataset._pydap_dataset
             .attributes['DODS_EXTRA']
             .update({'Unlimited_Dimension': 'index'}))
            dataset.dimensions = dataset._get_dims(dataset._pydap_dataset)
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
|S100 station(index)
    long_name: Station Name
unlimited dimensions: 
current shape = (4,)
"""
        with Dataset('http://localhost:8000/',
                     application=self.app) as dataset:
            variable = dataset.variables['station']
            assert repr(variable) == expected_repr

            # Mock unlimited dimension:
            assert not dataset.dimensions['index'].isunlimited()
            assert isinstance(dataset._pydap_dataset
                              .attributes['DODS_EXTRA'], dict)
            assert 'Unlimited_Dimension' not in (dataset
                                                 ._pydap_dataset
                                                 .attributes['DODS_EXTRA'])
            (dataset._pydap_dataset
             .attributes['DODS_EXTRA']
             .update({'Unlimited_Dimension': 'index'}))
            dataset.dimensions = dataset._get_dims(dataset._pydap_dataset)
            assert repr(variable) == (expected_repr
                                      .replace('unlimited dimensions: ',
                                               'unlimited dimensions: index'))

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
    url = ('http://cordexesg.dmi.dk/thredds/dodsC/cordex_general/'
           'cordex/output/EUR-11/DMI/ICHEC-EC-EARTH/historical/r3i1p1/'
           'DMI-HIRHAM5/v1/day/pr/v20131119/'
           'pr_EUR-11_ICHEC-EC-EARTH_historical_r3i1p1_'
           'DMI-HIRHAM5_v1_day_19960101-20001231.nc')
    test_url = url + '.dods?pr[0:1:0][0:1:5][0:1:5]'

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
        data = dataset['pr'][0, 200:205, 100:105]
        expected_data = [[[5.23546005e-05,  5.48864300e-05,
                           5.23546005e-05,  6.23914966e-05,
                           6.26627589e-05],
                          [5.45247385e-05,  5.67853021e-05,
                           5.90458621e-05,  6.51041701e-05,
                           6.23914966e-05],
                          [5.57906533e-05,  5.84129048e-05,
                           6.37478297e-05,  5.99500854e-05,
                           5.85033267e-05],
                          [5.44343166e-05,  5.45247385e-05,
                           5.60619228e-05,  5.58810752e-05,
                           4.91898136e-05],
                          [5.09982638e-05,  4.77430549e-05,
                           4.97323490e-05,  5.43438946e-05,
                           5.26258664e-05]]]
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
                          authentication_uri=esgf._uri(os.environ
                                                       .get('OPENID_ESGF')))
        data = dataset['pr'][0, 200:205, 100:105]
        expected_data = [[[5.23546005e-05,  5.48864300e-05,
                           5.23546005e-05,  6.23914966e-05,
                           6.26627589e-05],
                          [5.45247385e-05,  5.67853021e-05,
                           5.90458621e-05,  6.51041701e-05,
                           6.23914966e-05],
                          [5.57906533e-05,  5.84129048e-05,
                           6.37478297e-05,  5.99500854e-05,
                           5.85033267e-05],
                          [5.44343166e-05,  5.45247385e-05,
                           5.60619228e-05,  5.58810752e-05,
                           4.91898136e-05],
                          [5.09982638e-05,  4.77430549e-05,
                           4.97323490e-05,  5.43438946e-05,
                           5.26258664e-05]]]
        assert(np.isclose(data, expected_data).all())
