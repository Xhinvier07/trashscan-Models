# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import glob
import json
import ntpath
import os
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def unzip(file_path: str, dest_dir: str):
    """
    Unzips compressed .zip file.
    Example inputs:
        file_path: 'data/01_alb_id.zip'
        dest_dir: 'data/'
    """

    # unzip file
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dest_dir)


def save_json(data, save_path, indent: Optional[int] = None):
    """
    Saves json formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.json"
        indent: Train json files with indent=None, val json files with indent=4
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), cls=NumpyEncoder, indent=indent)


# type check when save json files
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def load_json(load_path: str, encoding: str = "utf-8"):
    """
    Loads json formatted data (given as "data") from load_path
    Encoding type can be specified with 'encoding' argument

    Example inputs:
        load_path: "dirname/coco.json"
    """
    # read from path
    with open(load_path, encoding=encoding) as json_file:
        data = json.load(json_file)
    return data


def list_files(
    directory: str,
    contains: list = [".json"],
    verbose: int = 1,
) -> List[str]:
    """
    Walk given directory and return a list of file path with desired extension

    Args:
        directory: str
            "data/coco/"
        contains: list
            A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
        verbose: int
            0: no print
            1: print number of files

    Returns:
        filepath_list : list
            List of file paths
    """
    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    filepath_list: List[str] = []

    for file in os.listdir(directory):
        # check if filename contains any of the terms given in contains list
        if any(strtocheck in file.lower() for strtocheck in contains):
            filepath = str(os.path.join(directory, file))
            filepath_list.append(filepath)

    number_of_files = len(filepath_list)
    folder_name = Path(directory).name

    verboseprint(f"There are {str(number_of_files)} listed files in folder: {folder_name}/")

    return filepath_list


def list_files_recursively(directory: str, contains: list = [".json"], verbose: bool = True) -> Tuple[list, list]:
    """
    Walk given directory recursively and return a list of file path with desired extension

    Arguments
    -------
        directory : str
            "data/coco/"
        contains : list
            A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
        verbose : bool
            If true, prints some results
    Returns
    -------
        relative_filepath_list : list
            List of file paths relative to given directory
        abs_filepath_list : list
            List of absolute file paths
    """

    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # walk directories recursively and find json files
    abs_filepath_list = []
    relative_filepath_list = []

    # r=root, d=directories, f=files
    for r, _, f in os.walk(directory):
        for file in f:
            # check if filename contains any of the terms given in contains list
            if any(strtocheck in file.lower() for strtocheck in contains):
                abs_filepath = os.path.join(r, file)
                abs_filepath_list.append(abs_filepath)
                relative_filepath = abs_filepath.split(directory)[-1]
                relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = directory.split(os.sep)[-1]

    verboseprint("There are {} listed files in folder {}.".format(number_of_files, folder_name))

    return relative_filepath_list, abs_filepath_list


def get_base_filename(path: str):
    """
    Takes a file path, returns (base_filename_with_extension, base_filename_without_extension)
    """
    base_filename_with_extension = ntpath.basename(path)
    base_filename_without_extension, _ = os.path.splitext(base_filename_with_extension)
    return base_filename_with_extension, base_filename_without_extension


def get_file_extension(path: str):
    """
    Get the file extension from a given file path.

    Args:
        path (str): The file path.

    Returns:
        str: The file extension.

    """
    _, file_extension = os.path.splitext(path)
    return file_extension


def load_pickle(load_path):
    """
    Loads pickle formatted data (given as "data") from load_path
    Example inputs:
        load_path: "dirname/coco.pickle"
    """
    with open(load_path, "rb") as json_file:
        data = pickle.load(json_file)
    return data


def save_pickle(data, save_path):
    """
    Saves pickle formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.pickle"
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "wb") as outfile:
        pickle.dump(data, outfile)


def import_model_class(model_type, class_name):
    """
    Imports a predefined detection class by class name.

    Args:
        model_type: str
            "yolov5", "detectron2", "mmdet", "huggingface" etc
        model_name: str
            Name of the detection model class (example: "MmdetDetectionModel")
    Returns:
        class_: class with given path
    """
    module = __import__(f"sahi.models.{model_type}", fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_


def increment_path(path: Union[str, Path], exist_ok: bool = True, sep: str = "") -> str:
    """
    Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.

    Args:
        path: str
            The base path to increment.
        exist_ok: bool
            If True, return the path as is if it already exists. If False, increment the path.
        sep: str
            The separator to use between the base path and the increment number.

    Returns:
        str: The incremented path.

    Example:
        >>> increment_path("runs/exp", sep="_")
        'runs/exp_0'
        >>> increment_path("runs/exp_0", sep="_")
        'runs/exp_1'
    """
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        indices = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(indices) + 1 if indices else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def download_from_url(from_url: str, to_path: str):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        from_url (str): The URL of the file to download.
        to_path (str): The path where the downloaded file should be saved.

    Returns:
        None
    """
    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )


def is_colab():
    """
    Check if the current environment is a Google Colab instance.

    Returns:
        bool: True if the environment is a Google Colab instance, False otherwise.
    """
    import sys

    return "google.colab" in sys.modules
