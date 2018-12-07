import platform
import os
from functools import partial
import collections


def ensure_executable(bin):
    if platform.system().lower() not in ('linux', 'darwin'):
        return
    import stat
    for ext in ['x86', 'x86_64']:
            filename = bin + '.' + ext
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)



class namedtuple(tuple):
    """An easier to use namedtuple. Example:

       class A(namedtuple):
          a = 4
          b = 1

       x = A(b=5)
       assert x == (4, 5) and x.b == 5 and x.a == 4 and x == A(4, 5)

    Note: this works because all dicts are ordered starting from python 3.6
    """
    def __init_subclass__(cls):
      annots = tuple(getattr(cls, '__annotations__', {}).keys())
      fields = tuple(n for n in vars(cls) if not n.startswith('__'))
      cls._nt = collections.namedtuple(cls.__name__, annots + fields)
      cls._nt.__new__.__defaults__ = tuple(vars(cls)[f] for f in fields)
    def __new__(cls, *args, **kwargs):
      return cls._nt(*args, **kwargs)


class A(namedtuple):
  a = 4
  b = None
  c: ...


x = A(b=5, c=8)
assert x == (8, 4, 5) and x.b == 5 and x.a == 4 and x == A(8, 4, 5)