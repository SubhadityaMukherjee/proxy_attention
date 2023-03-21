import pickle

def save_pickle(*args, fname="pickler.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(args, f)


def read_pickle(fname="pickler.pkl"):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj
