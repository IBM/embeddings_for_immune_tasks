import hashlib
import json
from itertools import tee
from typing import MutableMapping


def get_hash_int(data):
    """Generate a 64-bit integer"""
    data_str = hashlib.sha1(repr(data).encode()).hexdigest()[:8]
    return int(data_str, 16)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten_to_string(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_to_string(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_config(data, path):
    with open(path, "w") as jf:
        json.dump(data, jf)

def load_config(path):
    with open(path, "r") as jf:
        return json.load(jf)
