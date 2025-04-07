import requests
from .config import *
from typing import Dict, Union
from .types import DecoratorContextManager
from .std import *

oldRequest = requests.request
oldGet = requests.get
# oldPost = requests.post

useProxy = False
setProxies = config["proxy"]

def request(method, url, **kwargs):
    global setProxies
    print(f'request using proxy {config["proxy"]}')
    kwargs['proxies'] = setProxies
    return oldRequest(method, url, **kwargs)

def get(url, **kwargs):
    global setProxies
    print(f'GET using proxy {config["proxy"]}')
    kwargs['proxies'] = setProxies
    return oldGet(url, **kwargs)

def setRequestsProxy(proxies: Union[Dict[str, str], None]=None):
    '''
        设置 requests 代理，默认使用 config.json 中的 proxy
    '''
    global useProxy, setProxies
    setProxies = proxies if proxies != None else config["proxy"]
    # requests.request = request
    # requests.get = get
    os.environ['HTTP_PROXY'] = config['proxy']['http']
    os.environ['HTTPS_PROXY'] = config['proxy']['https']
    useProxy = True

def unsetRequestsProxy():
    '''
        设置 requests 不使用代理
    '''
    global useProxy
    # requests.request = oldRequest
    # requests.get = oldGet
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    useProxy = False



class RequestUsingProxy(DecoratorContextManager):
    '''
        上下文管理器 + 函数修饰器，requests 使用代理，默认使用 config.json 中的 proxy
    '''
    def __init__(self, proxies: Union[Dict[str, str], None]=None) -> None:
        '''
            上下文管理器 + 函数修饰器，requests 使用代理，默认使用 config.json 中的 proxy
        '''
        self.proxies = proxies
    
    def __enter__(self):
        setRequestsProxy(self.proxies)
        return self
    
    def __exit__(self, *args):
        unsetRequestsProxy()

@RequestUsingProxy()
def test():
    global useProxy
    print('======')
    print(useProxy)

if __name__ == '__main__':
    with RequestUsingProxy():
        print(useProxy)
    print(useProxy)
    test()
    print(useProxy)