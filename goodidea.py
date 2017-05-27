#!/usr/bin/python
""" this is an extremely good idea"""
import pandas as pd #technically just needed for tests
import numpy as np

def sloppy_vectorize(func, dframe, **kwargs):
    """ sloppily vectorizes a DataFrame-function pair, pushing any keywords
    that are also column names in the DataFrame in as vectors, then
    excludes all passed kwargs from vectorization while pushing any matches.
    returns a tuple of (return of vectorized, [unused cols],[bad kwargs])
    """
    func_kwords = func.__code__.co_varnames[:func.__code__.co_argcount]
    kwords = {}
    unused_cols = []
    for k, val in dframe.to_dict('series').items():
        if k in func_kwords:
            kwords[k] = val
        else:
            unused_cols.append(k)
    excluded_kw = []
    bad_kwargs = []
    for key, value in kwargs.items():
        if key in func_kwords:
            excluded_kw.append(key)
            kwords[key] = value
        else:
            bad_kwargs.append(key)

    vec_out = np.vectorize(func, excluded=excluded_kw)(**kwords)
    return vec_out, unused_cols, bad_kwargs


def test_sloppy_vectorize():
    """tests the above"""
    garbage_data = {
        'dog': [1, 1, 1, 2, 2],
        'cat': [2, 3, 4, 5, 6],
        'moose': [11, 12, 13, 15, 16],
    }

    dframe = pd.DataFrame.from_dict(garbage_data)
    test_kwargs = {
        'count': 2,
        'mount': 3,
        'flount': 4,
    }

    def crappy_function(dog, cat, count):
        """this sucks"""
        return dog * cat * count
    out = sloppy_vectorize(crappy_function, dframe, **test_kwargs)
    test_output = dframe['dog'] * dframe['cat'] * test_kwargs['count']
    for i in range(0, len(out[0])):
        assert out[0][i] == test_output[i]
    assert 'moose' in out[1]
    assert len(out[1]) == 1
    assert 'mount' in out[2]
    assert 'flount' in out[2]
    assert len(out[2]) == 2
