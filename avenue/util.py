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
        # if exists and is not executable try to make executable
        if os.path.isfile(filename) and not os.access(filename, os.X_OK):
          try:
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)
          except PermissionError:
            print(f"No permission to make {filename} executable")


def compute_assed_id(name, system):
    system = system.lower()
    assert system in ['windows', 'darwin', 'linux'], 'only windows, linux, mac are supported'
    path = '{}-{}'.format(name, system)
    return path


def compute_asset_path(asset_id):
    project_root = os.path.dirname(os.path.dirname(__file__))
    default_path = os.path.join(project_root, 'unity_assets')
    dir = os.environ.get('AVENUE_ASSETS', default_path)
    path = os.path.join(dir, asset_id)
    return path


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


def min_max_norm(x, min_value, max_value):
  return ((x - min_value) / (max_value - min_value) - 0.5) * 2


def test_namedtuple():
  class A(namedtuple):
    c: ...
    a = 4
    b = None
    

  x = A(b=5, c=8)
  assert x == (8, 4, 5) and x.b == 5 and x.a == 4 and x == A(8, 4, 5)


if __name__ == "__main__":
  test_namedtuple()



