import numpy as np


class PrintFormatter:
    def __init__(self):
        pass

    """
    Print matrix with specified scientific notation
    """

    def print_matrix(self, matrix: np.ndarray, name: str = "") -> None:
        with np.printoptions(
            precision=8, suppress=True, formatter={"float": "{:0.8e}".format}
        ):
            print(name)
            print(matrix)
