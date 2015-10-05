import time

class Timer(object):
    def __init__(self, name, print_func=print):
        self.name = name
        self.print_func = print_func

    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self,ty,val,tb):
        end = time.time()
        self.print_func("%s : %0.3f seconds" % (self.name, end-self.start))
        return False


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated"""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property