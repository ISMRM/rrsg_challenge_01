"""
This file contains misc functions that are used throughout the project...
"""


def print_dict(d, level=0):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, ':')
            print_dict(v, level + 1)
        else:
            print(level * '\t', k, '-', v)
