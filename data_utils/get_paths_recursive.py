"""
@version: 1.0
@summary: get paths of files
@Author: Salman Ahmed
@date: 2019-01-09
"""

import os


def get_file_paths(directory=".", max_depth=-1, file_ext='pkl'):
    """
    Recursively iterate over all nested directories and get paths of PKL files in directory and sub-directories.
    directory is PATH from where it will get all paths.
    max_depth is depth limit given as argument to how much deep we want to go.
    :param directory: directory to search for pkl file
    :param max_depth: num of sub directory  of current directory to search
    :param file_ext: ext of file
    :return:
    """

    if max_depth == 0:  # terminating condition for recursive function
        return []

    pkl_paths = []  # contains paths of all pkl files
    dirs = os.listdir(directory)  # get all files and directories in given directory

    for dir_ in dirs:  # iterate over files or directories in list
        if '.' not in dir_:  # a directory
            pkl_paths.extend(get_file_paths(directory=os.path.join(directory, dir_),
                                            max_depth=max_depth - 1, file_ext=file_ext))
        elif file_ext in dir_:  # a pkl file
            pkl_paths.append(os.path.join(directory, dir_))  # appending pkl file to lists
    return pkl_paths