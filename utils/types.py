from torch import no_grad

class DecoratorContextManager(no_grad.__base__):
    '''
        可以作为修饰器使用的上下文管理器类
    '''
    pass