from __future__ import annotations

import re
from collections import defaultdict
from functools import cached_property, lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import pandas as pd

from blacksquare.types import SpecialCellValue
from blacksquare.utils import sum_by_group

if TYPE_CHECKING:
    from blacksquare.word import Word


_ALPHA_REGEX = re.compile("^[A-Z]*$")


class ScoredWord(NamedTuple):
    word: str
    score: float


# letter_counts = pd.Series(
#   np.array(DEFAULT_WORDLIST.words).view('U1')
# ).value_counts().drop('')
# inverse_frequencies = (1/26) / (letter_counts/letter_counts.sum())
# inverse_frequencies.sort_index().to_dict()

INVERSE_CHARACTER_FREQUENCIES = {
    "A": 0.45,
    "B": 1.98,
    "C": 0.99,
    "D": 1.09,
    "E": 0.34,
    "F": 2.27,
    "G": 1.43,
    "H": 1.26,
    "I": 0.5,
    "J": 14.81,
    "K": 3.21,
    "L": 0.75,
    "M": 1.34,
    "N": 0.57,
    "O": 0.53,
    "P": 1.41,
    "Q": 27.79,
    "R": 0.56,
    "S": 0.51,
    "T": 0.55,
    "U": 1.19,
    "V": 3.76,
    "W": 3.18,
    "X": 10.08,
    "Y": 2.21,
    "Z": 12.62,
}


class WordList:
    def __init__(
        self,
        source: str | Path | list[str] | dict[str, int | float],
    ):
        """Representation of a scored word list.

        Args:
            source: The source for the word list. Can be a list of strings, a dict of
                strings to scores, a path to a .dict file with words in "word;score"
                format, or a path to a .npz file (produced to `.to_npz`) Words will be
                normalized and scores will be scaled from 0-1.

        Raises:
            ValueError: If input type is not recognized
        """
        if isinstance(source, str) or isinstance(source, Path):
            if Path(source).suffix == ".npz":
                loaded = np.load(source)
                length_keys = {
                    k.split("_")[0]
                    for k in loaded.keys()
                    if k not in ("words", "scores")
                }
                self._words = loaded["words"]
                self._scores = loaded["scores"]
                self._word_scores_by_length = {
                    int(k): (loaded[f"{k}_words"], loaded[f"{k}_scores"])
                    for k in length_keys
                }
                return
            else:
                df = pd.read_csv(
                    source,
                    sep=";",
                    header=None,
                    names=["word", "score"],
                    dtype={"word": str, "score": float},
                    na_filter=False,
                )
                raw_words_scores = df.values
        elif isinstance(source, list):
            assert len(source) > 0 and isinstance(source[0], str)
            raw_words_scores = [(w, 1) for w in source]
        elif isinstance(source, dict):
            raw_words_scores = list(source.items())
        else:
            raise ValueError("Input type not recognized")
        filtered_words_scores = [
            (_normalize(w), s)
            for w, s in raw_words_scores
            if _ALPHA_REGEX.match(_normalize(w))
        ]
        sorted_words_scores = sorted(filtered_words_scores, key=lambda x: (-x[1], x[0]))
        unzipped = list(zip(*sorted_words_scores))
        norm_words, scores = np.array(unzipped[0]), np.array(unzipped[1])
        norm_scores = scores / scores.max()

        self._words, self._scores = norm_words, norm_scores

        word_scores_by_length = defaultdict(lambda: ([], []))
        for word, score in zip(norm_words, norm_scores):
            word_scores_by_length[len(word)][0].append(word)
            word_scores_by_length[len(word)][1].append(score)

        self._word_scores_by_length = {
            length: (np.array(words, dtype=str), np.array(scores))
            for length, (words, scores) in word_scores_by_length.items()
        }

    def find_matches(self, word: Word) -> MatchWordList:
        """Find matches for a Word object.

        Args:
            word: The word to match.

        Returns:
            The matching words as a MatchWordList.
        """
        return self.find_matches_str(word.value)

    @lru_cache(128)
    def find_matches_str(self, query: str) -> MatchWordList:
        """Find matches for a query string. Open letters can be represented by a " ",
        "?", or "_" character.

        Args:
            query: The string to match against (e.g. "M???ING")

        Returns:
            A MatchWordList object containing the matching words.
        """
        query_array = np.array(list(query), dtype=str)
        if len(query_array) in self._word_scores_by_length:
            words, scores = self._word_scores_by_length[len(query_array)]

            empty_mask = np.any(
                [
                    query_array == repr_
                    for repr_ in SpecialCellValue.EMPTY.input_str_reprs
                ],
                axis=0,
            )
            letter_match_mask = query_array == words.view("U1").reshape(words.size, -1)
            match_mask = (letter_match_mask | empty_mask).all(axis=1)

            match_words, match_scores = words[match_mask], scores[match_mask]
            return MatchWordList(len(query_array), match_words, match_scores)
        else:
            return MatchWordList(
                len(query_array), np.empty((0,), dtype=str), np.empty((0,), dtype=float)
            )

    @cached_property
    def words(self) -> list[str]:
        return list(self._words)

    @cached_property
    def _words_dict(self) -> dict[str, float]:
        return dict(zip(self._words, self._scores))

    @cached_property
    def scores(self) -> list[float]:
        return list(self._scores)

    def get_score(self, word: str) -> float | None:
        """Return the score for a word.

        Args:
            word: The word to get the score for.

        Returns:
            The score. None if word is not in word list.
        """
        return self._words_dict.get(word)

    @cached_property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"word": self._words, "score": self._scores})

    def score_filter(self, threshold: float) -> WordList:
        """Returns a new word list containing only the words above the threshold.

        Args:
            threshold: The score threshold.

        Returns:
            The resulting WordList.
        """
        score_mask = self._scores >= threshold
        return WordList(dict(zip(self._words[score_mask], self._scores[score_mask])))

    def filter(self, filter_fn: Callable[[ScoredWord], bool]) -> WordList:
        """Returns a new word list filtered by a custom function.

        Args:
            filtern_fn: The filtering function. Takes a ScoredWord as an
                input and outputs a bool.

        Returns:
            The resulting WordList.
        """
        return WordList(dict([w for w in self if filter_fn(w)]))

    def to_npz(self, file: str | Path) -> None:
        """Serializes word list to a .npz format that is fast to load from disk.

        Args:
            file: The output file path.
        """
        by_length_arrays = {}
        for k, v in self._word_scores_by_length.items():
            by_length_arrays[f"{k}_words"] = v[0]
            by_length_arrays[f"{k}_scores"] = v[1]
        np.savez_compressed(
            file, words=self._words, scores=self._scores, **by_length_arrays
        )

    def __len__(self):
        return len(self._words)

    def __repr__(self):
        return f"WordList(\n{repr(self.frame)}\n)"

    def _repr_html_(self):
        return self.frame._repr_html_()

    def __getitem__(self, key) -> ScoredWord:
        if not isinstance(key, int):
            raise IndexError
        return ScoredWord(self._words[key], self._scores[key])

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            val = self[self._iter_index]
            self._iter_index += 1
            return val
        else:
            raise StopIteration

    def __add__(self, other):
        return WordList(
            {w: s for w, s in zip(self.words + other.words, self.scores + other.scores)}
        )

    def __contains__(self, item):
        return item in self._words_set


class MatchWordList(WordList):
    """An object representing a WordList matching an open word. This class should not be
    constructed by the user, it is generated by matching methods on the original word
    list.
    """

    def __init__(self, word_length: int, words: np.ndarray, scores: np.ndarray):
        self._words = words
        self._word_scores_by_length = {word_length: (words, scores)}
        self._word_length = word_length
        self._scores = scores

    def letter_scores_at_index(self, index: int) -> dict[str, int]:
        """The summed scores of matching letters at a given index.

        Args:
            index: The index to look at.

        Returns:
            A dict mapping letters to the summed scores of words containing that letter
            at the input index.
        """
        if len(self) > 0:
            letters = self._words.view("U1").reshape(self._words.size, -1)[:, index]
            return sum_by_group(letters, self._scores)
        else:
            return {}

    def rescore(
        self,
        rescore_fn: Callable[[str, float], float],
        drop_zeros=True,
    ) -> MatchWordList:
        """Generates a new word list with new scores, as defined by the rescore
        function.

        Args:
            rescore_fn: The function mapping the word and old score to new scores. This
                function should treat zero as invalid.
            drop_zeros: Whether to remove words with a score of zero. Defaults to True.

        Returns:
            A match word list where new scores are the results of the rescore function
            times the original score for the word.
        """
        vectorized_fn = np.vectorize(rescore_fn, otypes=[float])
        new_scores = vectorized_fn(self._words, self._scores)
        new_words = self._words
        if drop_zeros:
            gt_zero_mask = new_scores > 0
            new_scores = new_scores[gt_zero_mask]
            new_words = new_words[gt_zero_mask]

        sort_indices = np.flip(np.argsort(new_scores))
        return MatchWordList(
            self._word_length, new_words[sort_indices], new_scores[sort_indices]
        )

    def score_filter(self, threshold: float) -> MatchWordList:
        """Returns a new word list containing only the words above the threshold.

        Args:
            threshold: The score threshold.

        Returns:
            The resulting MatchWordList
        """
        score_mask = self._scores >= threshold
        return MatchWordList(
            self._word_length, self._words[score_mask], self._scores[score_mask]
        )

    def filter_words(self, words: list[str]) -> MatchWordList:
        """Returns a new word list with a specific set of words filtered out.

        Args:
            words: The list of words to filter out.

        Returns:
            The new MatchWordlist.
        """
        word_mask = ~np.isin(self._words, words)
        return MatchWordList(
            self._word_length, self._words[word_mask], self._scores[word_mask]
        )


def _normalize(word: str) -> str:
    """Sanitizes an input word.

    Args:
        word: The input word.

    Returns:
        An upper-cased trimmed string.
    """
    return word.upper().replace(" ", "")


DEFAULT_WORDLIST = WordList(files("blacksquare").joinpath("spreadthewordlist.npz"))
