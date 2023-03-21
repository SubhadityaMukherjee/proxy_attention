import mimetypes
from pathlib import Path
from typing import Generator, Iterable
import pickle
import seaborn as sns

sns.set()
import os

# from fast.ai source code
image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable, Generator)) and getattr(o, "ndim", 1)


def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, "__len__") and getattr(o, "ndim", 1)


def is_array(x):
    "`True` if `x` supports `__array__` or `iloc`"
    return hasattr(x, "__array__") or hasattr(x, "iloc")


def listify(o=None, *rest, use_list=False, match=None):
    "Convert `o` to a `list`"
    if rest:
        o = (o,) + rest
    if use_list:
        res = list(o)
    elif o is None:
        res = []
    elif isinstance(o, list):
        res = o
    elif isinstance(o, str) or is_array(o):
        res = [o]
    elif is_iter(o):
        res = list(o)
    else:
        res = [o]
    if match is not None:
        if is_coll(match):
            match = len(match)
        if len(res) == 1:
            res = res * match
        else:
            assert len(res) == match, "Match length mismatch"
    return res


def setify(o):
    "Turn any list like-object into a set."
    return o if isinstance(o, set) else set(listify(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders = []
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res

def save_pickle(*args, fname = "pickler.pkl"):
    with open(fname, 'wb') as f:
        pickle.dump(args, f)

def read_pickle(fname = "pickler.pkl"):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj