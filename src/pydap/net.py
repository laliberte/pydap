from requests.exceptions import MissingSchema
from requests.packages.urllib3.exceptions import (InsecureRequestWarning,
                                                  InsecurePlatformWarning)

from webob.request import Request
from webob.exc import HTTPError
from contextlib import closing
import ssl
import warnings

from six.moves.urllib.parse import urlsplit, urlunsplit

from .lib import DEFAULT_TIMEOUT

ssl_verify_categories = [InsecureRequestWarning,
                         InsecurePlatformWarning]


def GET(url, application=None, session=None,
        timeout=DEFAULT_TIMEOUT, verify=True):
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
                           timeout=timeout, verify=verify)


def raise_for_status(response):
    if response.status_code >= 400:
        raise HTTPError(
            detail=response.status+'\n'+response.text,
            headers=response.headers,
            comment=response.body
        )
    elif response.status_code >= 300:
        try:
            text = response.text
        except AttributeError:
            # With this status_code, response.text could
            # be ill-defined. If the redirect does not set
            # an encoding (i.e. response.charset is None).
            # Set the text to empty string:
            text = ''
        raise HTTPError(
            detail=(response.status + '\n' + text + '\n' +
                    'This is redirect error. These should not usually raise ' +
                    'an error in pydap beacuse redirects are handled ' +
                    'implicitly. If it failed it is likely due to a ' +
                    'circular redirect.'),
            headers=response.headers,
            comment=response.body)


def follow_redirect(url, application=None, session=None,
                    timeout=DEFAULT_TIMEOUT, verify=True):
    """
    This function essentially performs the following command:
    >>> Request.blank(url).get_response(application)

    It however makes sure that the request possesses the same cookies and
    headers as the passed session.
    """
    req = create_request(url, session=session, verify=verify,
                         timeout=timeout)
    req.environ['webob.client.timeout'] = timeout
    return get_response(req, application, verify=verify)


def get_response(req, application, verify=True):
    """
    If verify=False, use the ssl library to temporarily disable
    ssl verification.
    """
    if verify:
        resp = req.get_response(application)
    else:
        # Here, we use monkeypatching. Webob does not provide a way
        # to bypass SSL verification.
        # This approach is never ideal but it appears to be the only option
        # here.
        # This only works in python 2.7 and >=3.5. In python 3.3 and 3.4
        # do not require it because by default contexts are not
        # verified.
        try:
            _create_default_https_ctx = ssl._create_default_https_context
            _create_unverified_ctx = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_ctx
        except AttributeError:
            _create_default_https_ctx = None

        try:
            resp = req.get_response(application)
        finally:
            if _create_default_https_ctx is not None:
                # Restore verified context
                ssl._create_default_https_context = _create_default_https_ctx
    return resp


def create_request(url, session=None, verify=True,
                   timeout=DEFAULT_TIMEOUT):
    """
    By default, create a blank request. If session
    is available, follow redirects.
    """
    if session is None:
        return Request.blank(url)

    # Use session to follow redirects:
    with warnings.catch_warnings():
        if not verify:
            # Catch warnings. It is assumed that the
            # user that explicitly uses verify=False
            # is either fully aware of the risks
            # or cannot avoid the risks because of
            # an improperly configured server.
            # This error will usually occur with
            # ESGF authentication.
            for category in ssl_verify_categories:
                warnings.filterwarnings("ignore",
                                        category=category)
        try:
            with closing(session.head(url, allow_redirects=True,
                                      verify=verify,
                                      timeout=timeout)) as head:
                req = Request.blank(head.url)

                # Get cookies from head:
                cookies_dict = head.cookies.get_dict()

                # Set request cookies to the head cookies:
                req.headers['Cookie'] = ','.join(name + '=' +
                                                 cookies_dict[name]
                                                 for name in cookies_dict)
                # Set the headers to the session headers:
                for item in head.request.headers:
                    req.headers[item] = head.request.headers[item]
                return req
        except MissingSchema:
            # In testing, missing schema means that
            # a arbitrary url was used. Simply
            # pass
            return Request.blank(url)
