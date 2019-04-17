"""
Module py_utils

Miscellaneous utilities for python3
"""
from functools import reduce
import operator

import numpy as np


def xyz2mesh(ara):
    """
    input ar: A numpy array with columns x, y, z such that the (x,y) pairs
    form a complete grid.  The rows of A may be in any order.

    output: 3 numpy arrays X, Y, Z in meshgrid form, suitable for constructing
    a 3d mesh plot or contour map.
    """
    nx, ny = (len(np.unique(ara[:, i])) for i in range(2))
    arb = [tuple(r) for r in list(ara)]
    arb.sort(key=lambda t: (t[1], t[0]))
    arc = np.array(arb)
    return tuple(np.reshape(arc[:, i], (ny, nx)) for i in range(3))


def mesh2xyz(arx, ary, arz):
    """
    inputs arx, ary, arz: 3 numpy arrays in meshgrid form.

    output: A numpy array with columns x, y, z such that the (x,y) pairs
    form a complete grid.
    """
    return np.vstack((arx.ravel(), ary.ravel(), arz.ravel())).T


def multidim2table(ar, *dim_vals, titles=None, widths=None):
    """
    Convert a multidimensional array to a string representation of
    a table of values.

    positional arguments
    ar: a multidimensional numpy array
    dim_vals: collections enumerating the basis for each dimension,
        typically the values for a given input in a trial

    keyword arguments
    titles: a list of column titles, each corresponding to a collection
        in dim_vals
    widths: a list of column widths, a width or None
    """
    return list2table(multidim2list(ar, *dim_vals, titles=titles),
                      widths=widths)


def multidim2list(ar, *dim_vals, titles=None):
    """
    Converts a multidimensional array to a list of tuples
    This may be useful for displaying part or all of an array as a
    table or exporting to a pandas dataframe, database or spreadsheet.

    positional arguments
    ar: a multidimensional numpy array
    dim_vals: collections enumerating the basis for each dimension,
        typically the values for a given input in a trial

    keyword arguments
    titles: a list of column titles, each corresponding to a collection
        in dim_vals, except the last which refers to the value.  If the
        last title is omitted, 'value' is used.
    """
    def filled_list(idx):
        """
        adds appropriate copies of a list item based on its index within
        the dim_vals list
        """
        shape = [len(dv) for dv in dim_vals]
        lst = dim_vals[idx]
        pre = shape[:idx]
        post = shape[idx+1:]
        preprod = reduce(operator.mul, pre, 1)
        postprod = reduce(operator.mul, post, 1)
        result = []
        for item in lst:
            result += [item] * postprod
        result *= preprod
        return result

    cols = []
    for i in range(len(dim_vals)):
        cols.append(filled_list(i))
    cols.append(list(ar.ravel()))
    result = list(zip(*cols))
    if titles:
        if len(titles) == len(result[0]) - 1:
            titles.append('value')
        result.insert(0, tuple(titles))
    return result

def multidim2cmptable(ar1, ar2, *dim_vals, titles=None, widths=None):
    """
    Convert two multidimensional arrays of the same shape
    to a string representation of a table of values representing a 
    comparison of the values in the arrays.

    positional arguments
    ar: a multidimensional numpy array
    dim_vals: collections enumerating the basis for each dimension,
        typically the values for a given input in a trial

    keyword arguments
    titles: a list of column titles, each corresponding to a collection
        in dim_vals
    widths: a list of column widths, a width or None
    """
    return list2table(multidim2cmplist(ar1, ar2, *dim_vals, titles=titles),
                      widths=widths)


def multidim2cmplist(ar1, ar2, *dim_vals, titles=None):
    """
    Converts a multidimensional array to a list of tuples
    This may be useful for displaying part or all of an array as a
    table or exporting to a pandas dataframe, database or spreadsheet.

    positional arguments
    ar: a multidimensional numpy array
    dim_vals: collections enumerating the basis for each dimension,
        typically the values for a given input in a trial

    keyword arguments
    titles: a list of column titles, each corresponding to a collection
        in dim_vals, except the last which refers to the value.  If the
        last title is omitted, 'value' is used.
    """
    def filled_list(idx):
        """
        adds appropriate copies of a list item based on its index within
        the dim_vals list
        """
        shape = [len(dv) for dv in dim_vals]
        lst = dim_vals[idx]
        pre = shape[:idx]
        post = shape[idx+1:]
        preprod = reduce(operator.mul, pre, 1)
        postprod = reduce(operator.mul, post, 1)
        result = []
        for item in lst:
            result += [item] * postprod
        result *= preprod
        return result

    cols = []
    for i in range(len(dim_vals)):
        cols.append(filled_list(i))
    ar3 = ar1 / ar2
    cols.append(["%1.2e" % v for v in list(ar1.ravel())])
    cols.append(["%1.2e" % v for v in list(ar2.ravel())])
    cols.append(["%1.2e" % v for v in list(ar3.ravel())])
    result = list(zip(*cols))
    if titles:
        if len(titles) == len(result[0]) - 2:
            titles.append('value1', 'value2')
        result.insert(0, tuple(titles))
    return result



def list2table(lst, widths=None):
    """
    Converts a list of tuples to a table represented as a string

    positional arguments
    lst: the list of tuples

    keyword arguments
    widths: a list of individual column widths, a single number or
        None (the default).  If None is received, a width of 10 is used
    """
    strlist = []
    if widths is None:
        widths = [10]*len(lst[0])
    elif getattr(widths, "__len__", None) is None:
        widths = [widths]*len(lst[0])
    for row in lst:
        strlist.append(''.join([str(item).ljust(widths[i]) for i, item
                                in enumerate(row)]))
    return '\n'.join(strlist)

def csv2pretty(csvstr, sep='\t'):
    """
    Format CSV text with the specified separator to 'pretty' text
    """
    def formatted_line(words, maxes, pad):
        st = ''
        for j, item in enumerate(words):
            st += item.rjust(maxes[j] + pad)
        return st + '\n'

    result = '\n'
    lines = csvstr.strip().split('\n')
    items = [line.split(sep) for line in lines]
    cols = list(zip(*items))
    maxlens = [max([len(word) for word in col]) for col in cols]
    pad = 4
    width = sum(maxlens) + len(cols)*pad
    result += '-'*width + '\n'
    result += formatted_line(items[0], maxlens, pad)
    result += '-'*width + '\n'
    for i, itm in enumerate(items[1:]):
        result += formatted_line(itm, maxlens, pad)
    result += '-'*width + '\n'
    return result


if __name__ == '__main__':
    X, Y = np.meshgrid(np.arange(4), np.arange(6))
    Z = X*Y
    A = mesh2xyz(X, Y, Z)
    # mess up the sorting for test purposes.
    B = [tuple(r) for r in list(A)]
    B.sort(key=lambda t: t[2])
    C = np.array(B)
    X1, Y1, Z1 = xyz2mesh(C)
    for (m1, m2) in zip([X, Y, Z], [X1, Y1, Z1]):
        assert np.all(m1 - m2 == 0)

    # test multidim2list
    FICTLS = [0.001, 0.1]
    PERCENTS = [10, 50]
    PUMP_POWERS = [1000, 2500, 5000]
    MODES = ['LP01', 'LP11']
    COLHEADS = ['fictL', '%LP01', 'pump', 'mode', 'error']

    A = np.array([[[[1, 2], [3, 4], [5, 6]], [[2, 1], [4, 3], [6, 5]]],
                  [[[1, 1], [3, 3], [5, 5]], [[2, 2], [4, 4], [6, 6]]]])
    MYLIST = multidim2list(A, FICTLS, PERCENTS, PUMP_POWERS,
                           MODES, titles=COLHEADS)
    print(list2table(MYLIST))


