from pathlib import Path
from typing import Union, Dict, List
import csv
import numpy as np
import re
from functools import lru_cache
from frozendict import frozendict


_DEFAULT_DICT_PATH = "peter-broda-wordlist__score_20210102.txt"


Dictionary = Union[Dict[str, float], List[str], frozendict]
_ALPHA_REGEX = re.compile("^[A-Z]*$")


def load_default_dictionary() -> frozendict:
    """Loads the default dictionary (Peter Broda's scored wordlist) in the proper
    frozendict format.

    Returns:
        frozendict: The loaded frozendict.
    """
    path = Path(__file__).parent / _DEFAULT_DICT_PATH
    with path.open(encoding="ISO-8859-1", newline="") as f:
        dictionary = {
            w.upper(): int(s) / 100
            for w, s in csv.reader(f, delimiter=";")
            if _ALPHA_REGEX.match(w.upper())
        }
    return frozendict(sorted(dictionary.items()))


def parse_dictionary(input: Dictionary) -> frozendict:
    """Helper method for taking an input dictionary and normalizing words and scores.
    Also converts input types to a frozendict so that matching queries can be cached.

    Args:
        input (Dictionary): An input dictionary. Either a list of strings, or a dict
            mapping strings to positive numeric scores.

    Returns:
        frozendict: The normalized dictionary. All words will be uppercased with spaces
            removed. All scores will be between 0 and 1.
    """

    def normalize(string: str) -> str:
        return string.upper().replace(" ", "")

    if isinstance(input, list):
        dictionary = {
            normalize(w): 1.0 for w in input if _ALPHA_REGEX.match(normalize(w))
        }
    elif isinstance(input, dict):
        max_val = max(input.values())
        dictionary = {normalize(w): v / max_val for w, v in input.items()}
    elif isinstance(input, frozendict):
        return input
    return frozendict(sorted(dictionary.items()))


@lru_cache(maxsize=1024)
def cached_regex_match(dictionary: frozendict, regex_string: str) -> np.ndarray:
    """A method for regex matching to a dictionary's keys that uses caching to improve
    speed.

    Args:
        dictionary (frozendict): The dictionary to search.
        regex_string (str): The regular expression to match.

    Returns:
        np.ndarray: A numpy array of strings in the dictionary that match the regex.
    """
    rex = re.compile(regex_string)
    return np.array(list(filter(rex.match, dictionary.keys())))