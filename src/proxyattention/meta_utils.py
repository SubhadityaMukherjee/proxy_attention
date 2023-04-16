import mimetypes
from pathlib import Path
from typing import Generator, Iterable
import pickle
import seaborn as sns
import clipboard
import pandas as pd

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


def save_pickle(*args, fname="pickler.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(args, f)


def read_pickle(fname="pickler.pkl"):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj

def check_proxy(string): return "p" in str(string)
def calc_stats(values):
    return f"min: {values.min()} \nmax: {values.max()} \navg: {values.mean()}"
def convert_float(df, cols, totype= float):
    for col in cols:
        df[col] = df[col].astype(totype)

def fix_tensorboard_names(df)->pd.DataFrame:
    """Fixes the names of the columns in the tensorboard csv file."""
    df = df[df["global_run_count"].isna() == False]

    convert_float(df, ["global_run_count"], int)
    df = df.fillna(0)
    # Fix naming
    df = df.rename(columns={"Acc/Val":"accuracy", "proxy_steps":"step_schedule"})
    # Fix types
    convert_float(df, ["change_subset_attention", "proxy_threshold", "accuracy"], float)
    convert_float(df, ["transfer_imagenet"], bool)

    df["has_proxy"] = df["step_schedule"].apply(check_proxy)
    return df


def return_grouped_results(df, group_cols ,filter = None, index_cols = (["ds_name", ("accuracy")]), print_latex = False):

    if filter != None:
        df = df.reset_index()
        for key in filter.keys():
            df = df[df[key] == filter[key]]
    final_df = pd.DataFrame(df.groupby(group_cols, as_index=True).mean(numeric_only = True)["accuracy"]).sort_values(index_cols, ascending=False)
    if print_latex == True:
        clipboard.copy(final_df.to_latex())

    return final_df

