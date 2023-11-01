import cv2
import yaml


class YamlReader:
    def __init__(self) -> None:
        def opencv_matrix_constructor(loader, node):
            value = loader.construct_mapping(node, deep=True)
            return value

        yaml.SafeLoader.add_constructor(
            "tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor
        )

    """
    Read matrix related to letter from given file
    """

    def read_file(self, file_path, pass_first_line=False) -> dict:
        with open(file_path, "r") as file:
            if pass_first_line:
                file.readline()

            return yaml.safe_load(file)
