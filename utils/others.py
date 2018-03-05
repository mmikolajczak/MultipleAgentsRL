import re
import os


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_path_to_file_last_in_numerical_order(dir_path):
    return os.path.join(dir_path, next(reversed(natural_sort(os.listdir(dir_path)))))
