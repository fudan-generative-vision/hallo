"""
This module provides utility functions for configuration manipulation.
"""

from typing import Dict


def filter_non_none(dict_obj: Dict):
    """
    Filters out key-value pairs from the given dictionary where the value is None.

    Args:
        dict_obj (Dict): The dictionary to be filtered.

    Returns:
        Dict: The dictionary with key-value pairs removed where the value was None.

    This function creates a new dictionary containing only the key-value pairs from
    the original dictionary where the value is not None. It then clears the original
    dictionary and updates it with the filtered key-value pairs.
    """
    non_none_filter = { k: v for k, v in dict_obj.items() if v is not None }
    dict_obj.clear()
    dict_obj.update(non_none_filter)
    return dict_obj
