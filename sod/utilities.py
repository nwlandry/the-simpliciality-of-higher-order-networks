import numpy as np


def truncated_power_law(n, minval, maxval, r):
    u = np.random.random(n)
    a = minval ** (1 - r)
    b = maxval ** (1 - r)
    return {i: int(val) for i, val in enumerate((a + u * (b - a)) ** (1 / (1 - r)))}


def truncated_power_law_mean(minval, maxval, r):
    a = minval ** (2 - r)
    b = maxval ** (2 - r)
    c = minval ** (1 - r)
    d = maxval ** (1 - r)
    return (r - 1) / (r - 2) * (a - b) / (c - d)


def list_of_lists_to_latex_table(data, column_labels):
    """list of lists and column labels to latex table

    Parameters
    ----------
    data : list of lists
        list of table rows
    column_labels : list of str
        columns labels

    Raises
    ------
    Exception
        If the dimensions don't match.
    """

    if len(data[0]) != len(column_labels):
        raise Exception("Column labels and data entries must have the same length!")

    m = len(column_labels)
    print("\\begin{table}")
    print("\\begin{center}")
    print("\\begin{tabular}{" + "c" * m + "}")
    print(" & ".join([str(e) for e in column_labels]) + " \\\ ")
    print("\\hline")
    for row in data:
        print(" & ".join([str(e) for e in row]) + " \\\ ")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")
