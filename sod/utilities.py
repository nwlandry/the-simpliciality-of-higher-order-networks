import numpy as np


def list_of_lists_to_latex_table(data, column_labels, decimals=2):
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
        print(
            " & ".join(
                [str(round(e, decimals) if isinstance(e, float) else e) for e in row]
            )
            + " \\\ "
        )
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")
