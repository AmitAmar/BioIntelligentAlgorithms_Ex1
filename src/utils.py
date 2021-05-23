import os


def create_output_dir(dir_path):
    """
    Creates the model's directory if it doesn't exists
    :param dir_path: directory path
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
