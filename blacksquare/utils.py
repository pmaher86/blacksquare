import numpy as np
from typing import Dict


def sum_by_group(groups: np.ndarray, values: np.ndarray) -> Dict:
    """A utility function to perform grouped summing of values over numpy arrays.

    Args:
        groups (np.ndarray): An array describing what group each value corresponds to.
        values (np.ndarray): An array of values.

    Returns:
        Dict: A dictionary mapping each unique group the sum of its values.
    """
    sort_indices = np.argsort(groups)
    sorted_groups = groups[sort_indices]
    sorted_values = values[sort_indices]
    unique_groups, unique_indices = np.unique(sorted_groups, return_index=True)
    group_values = np.split(sorted_values, unique_indices[1:])
    return {group: values.sum() for group, values in zip(unique_groups, group_values)}
