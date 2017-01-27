from .lib import DEFAULT_TIMEOUT
import sys
import warnings

from webob.request import Request
from webob.exc import HTTPError
from contextlib import closing

from six.moves.urllib.parse import urlsplit, urlunsplit


def GET(url, application=None, session=None,
        timeout=DEFAULT_TIMEOUT):
    """Open a remote URL returning a webob.response.Response object

    Optional parameters:
    session: a requests.Session() object (potentially) containing
             authentication cookies.

    Optionally open a URL to a local WSGI application
    """
    if application:
        _, _, path, query, fragment = urlsplit(url)
        url = urlunsplit(('', '', path, query, fragment))

    return follow_redirect(url, application=application, session=session,
                           timeout=timeout)


def raise_for_status(response):
    if response.status_code >= 400:
        raise HTTPError(
            detail=response.status+'\n'+response.text,
            headers=response.headers,
            comment=response.body
        )


def follow_redirect(url, application=None, session=None,
                    timeout=DEFAULT_TIMEOUT,
                    cookies_dict=None):
    """
    This function essentially performs the following command:
    >>> Request.blank(url).get_response(application)

    It however makes sure that the request possesses the same cookies and
    headers as the passed session.
    """

    if session is not None:
        # Use session to follow redirects:
        with closing(session.head(url)) as head:
            req = Request.blank(head.url)

            # Get cookies from head:
            if cookies_dict is None:
                cookies_dict = head.cookies.get_dict()

            # Set request cookies to the head cookies:
            req.headers['Cookie'] = ','.join(name + '=' + cookies_dict[name]
                                             for name in cookies_dict)
            # Set the headers to the session headers:
            for item in head.request.headers:
                req.headers[item] = head.request.headers[item]
    else:
        # Use session to follow redirects:
        req = Request.blank(url)

    if (timeout != DEFAULT_TIMEOUT and
       sys.version_info < (2, 7)):
        warnings.warn('Currently pydap does not support '
                      'user-specified timeouts in python 2.6',
                      DeprecationWarning)
    req.environ['webob.client.timeout'] = timeout
    return req.get_response(application)
