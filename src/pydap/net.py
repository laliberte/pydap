from .lib import DEFAULT_TIMEOUT
from requests.exceptions import MissingSchema

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
                    timeout=DEFAULT_TIMEOUT):
    """
    This function essentially performs the following command:
    >>> Request.blank(url).get_response(application)

    It however makes sure that the request possesses the same cookies and
    headers as the passed session.
    """
    req = create_request(url, session=session)
    req.environ['webob.client.timeout'] = timeout
    return req.get_response(application)


def create_request(url, session=None):
    if session is not None:
        try:
            # Use session to follow redirects:
            with closing(session.head(url)) as head:
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
            pass
    return Request.blank(url)
