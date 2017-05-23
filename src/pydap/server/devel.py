from webob.request import Request
from webob.exc import HTTPError
import threading
import multiprocessing
import time
import math
import numpy as np
import socket

from wsgiref.simple_server import make_server

from ..handlers.lib import BaseHandler
from ..model import BaseType, DatasetType
from ..wsgi.ssf import ServerSideFunctions

DefaultDataset = DatasetType("Default")
DefaultDataset["byte"] = BaseType("byte", np.arange(5, dtype="B"))
DefaultDataset["string"] = BaseType("string", np.array(["one", "two"]))
DefaultDataset["short"] = BaseType("short", np.array(1, dtype="h"))


def get_open_port():
    # http://stackoverflow.com/questions/2838244/
    # get-open-tcp-port-in-python/2838309#2838309
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def run_server_in_process(httpd, shutdown, **kwargs):
    _server = (threading
               .Thread(target=httpd.serve_forever,
                       kwargs=kwargs))
    _server.start()
    shutdown.wait()
    httpd.shutdown()
    _server.join()


def shutdown_application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Server is shutting down.']


class LocalTestServer:
    """
    Simple server instance that can be used to test pydap.
    Relies on threading and is usually slow (it has to
    start and shutdown which typically takes ~2 sec).

    Usage:
    >>> import numpy as np
    >>> from pydap.handlers.lib import BaseHandler
    >>> from pydap.model import DatasetType, BaseType
    >>> DefaultDataset = DatasetType("Default")
    >>> DefaultDataset["byte"] = BaseType("byte", np.arange(5, dtype="B"))
    >>> DefaultDataset["string"] = BaseType("string", np.array(["one", "two"]))
    >>> DefaultDataset["short"] = BaseType("short", np.array(1, dtype="h"))
    >>> DefaultDataset
    <DatasetType with children 'byte', 'string', 'short'>
    >>> application = BaseHandler(DefaultDataset)
    >>> from pydap.client import open_url

    As an instance:
    >>> with LocalTestServer(application) as server:
    ...     dataset = open_url("http://localhost:%s" % server.port)
    ...     dataset
    ...     print(dataset['byte'].data[:])
    ...     print(dataset['string'].data[:])
    ...     print(dataset['short'].data[:])
    <DatasetType with children 'byte', 'string', 'short'>
    [0 1 2 3 4]
    [b'one' b'two']
    1

    Or by managing connection and deconnection:
    >>> server = LocalTestServer(application)
    >>> server.start()
    >>> dataset = open_url("http://localhost:%s" % server.port)
    >>> dataset
    <DatasetType with children 'byte', 'string', 'short'>
    >>> print(dataset['byte'].data[:])
    [0 1 2 3 4]
    >>> server.shutdown()
    """

    def __init__(self, application=BaseHandler(DefaultDataset),
                 port=None, wait=0.5, polling=1e-2, as_process=False,
                 ssl_context=None):
        self._port = port or get_open_port()
        self.application = application
        self._wait = wait
        self._polling = polling
        self._as_process = as_process
        self._ssl_context = ssl_context

    def start(self):
        # Start a simple WSGI server:
        application = ServerSideFunctions(self.application)
        address = '0.0.0.0'
        if self._ssl_context is None:
            self._httpd = make_server(address, self.port, application)
        else:
            from werkzeug.serving import make_server as make_server_ssl
            self._httpd = make_server_ssl(address, self.port, application,
                                          **{'ssl_context': self._ssl_context})
        self.url = "http://{0}:{1}/".format(address, self.port)

        if self._as_process:
            self._shutdown = multiprocessing.Event()
            self._server = (multiprocessing
                            .Process(target=run_server_in_process,
                                     args=(self._httpd, self._shutdown),
                                     kwargs={'poll_interval': 0.1}))
        else:
            self._server = (threading
                            .Thread(target=self._httpd.serve_forever,
                                    kwargs={'poll_interval': 0.1}))

        self._server.start()

        # Poll the server
        ok = False
        for trial in range(int(math.ceil(self._wait/self._polling))):
            try:
                resp = (Request
                        .blank("http://0.0.0.0:%s/.dds" % self.port)
                        .get_response())
                ok = (resp.status_code == 200)
            except HTTPError:
                pass
            if ok:
                break
            time.sleep(self._polling)

        if not ok:
            raise Exception(('LocalTestServer did not start in {0}s. '
                             'Try using LocalTestServer(..., wait={1}')
                            .format(self._wait, 2*self._wait))

    @property
    def port(self):
        return self._port

    def __enter__(self):
        self.start()
        return self

    def shutdown(self):
        # Tell the server to shutdown:
        if self._as_process:
            self._shutdown.set()
        else:
            self._httpd.shutdown()
        self._server.join()
        self._httpd.server_close()

    def __exit__(self, *_):
        self.shutdown()
