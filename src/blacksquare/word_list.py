from __future__ import annotations

import re
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:
    from importlib.abc import Traversable

import numpy as np
import pandas as pd

from blacksquare._blacksquare_rs import (
    PyMatchWordList,
    PyWordList,
    get_inverse_character_frequencies,
)

if TYPE_CHECKING:
    from blacksquare.word import Word


_ALPHA_REGEX = re.compile("^[A-Z]*$")


class ScoredWord(NamedTuple):
    word: str
    score: float


INVERSE_CHARACTER_FREQUENCIES = get_inverse_character_frequencies()


class WordList:
    _inner: PyWordList

    def __init__(
        self,
        source: str
        | Path
        | Traversable
        | list[str]
        | dict[str, int | float]
        | PyWordList
        | None = None,
    ):
        """Representation of a scored word list backed by an ultra-fast Rust engine.

        Args:
            source: The source for the word list. Can be a list of strings, a dict of
                strings to scores, a path to a .dict file with words in "word;score"
                format, or a path to a .npz file (produced to `.to_npz`) Words will be
                normalized and scores will be scaled from 0-1. If None, the built-in
                embedded default dictionary is used.

        Raises:
            ValueError: If input type is not recognized
        """
        if source is None:
            self._inner = PyWordList.default()
        elif isinstance(source, PyWordList):
            self._inner = source
        elif isinstance(source, (str, Path, Traversable)):
            path = Path(str(source))
            if path.suffix == ".npz":
                loaded = np.load(path)
                length_keys = {
                    k.split("_")[0]
                    for k in loaded.keys()
                    if k not in ("words", "scores")
                }
                words_list = [str(w) for w in loaded["words"]]
                scores_list = [float(s) for s in loaded["scores"]]
                self._words_cache = np.asarray(loaded["words"], dtype=str)
                self._scores_cache = np.asarray(loaded["scores"], dtype=float)
                self._by_length_cache = {
                    int(k): (
                        np.asarray(loaded[f"{k}_words"], dtype=str),
                        np.asarray(loaded[f"{k}_scores"], dtype=float),
                    )
                    for k in length_keys
                }
                self._inner = PyWordList.from_words_scores(words_list, scores_list)
                return
            else:
                self._inner = PyWordList(str(path))
        elif isinstance(source, list):
            assert len(source) > 0 and isinstance(source[0], str)
            self._inner = PyWordList(source)
        elif isinstance(source, dict):
            self._inner = PyWordList({str(k): float(v) for k, v in source.items()})
        else:
            raise ValueError("Input type not recognized")

        self._words_cache: np.ndarray | None = None
        self._scores_cache: np.ndarray | None = None
        self._by_length_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def _words(self) -> np.ndarray:
        if self._words_cache is None:
            self._words_cache = np.asarray(self._inner.words, dtype=str)
        return self._words_cache

    @property
    def _scores(self) -> np.ndarray:
        if self._scores_cache is None:
            self._scores_cache = np.asarray(self._inner.scores, dtype=float)
        return self._scores_cache

    @property
    def _word_scores_by_length(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        if self._by_length_cache is None:
            res: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for length in range(1, 32):
                part = self._inner.get_partition(length)
                if part is not None and len(part[0]) > 0:
                    res[length] = (
                        np.asarray(part[0], dtype=str),
                        np.asarray(part[1], dtype=float),
                    )
            self._by_length_cache = res
        return self._by_length_cache

    @property
    def words(self) -> list[str]:
        return self._inner.words

    @property
    def scores(self) -> list[float]:
        return self._inner.scores

    def get_score(self, word: str) -> float | None:
        """Looks up the score for a single word.

        Args:
            word: The word to look up.

        Returns:
            The score of the word, or None if the word is not in the list.
        """
        return self._inner.get_score(word)

    def find_matches(self, word: Word | str) -> MatchWordList:
        """Finds all words in the word list that match the input word pattern.

        Args:
            word: The word to find matches for. Can be a string containing wildcards
                (e.g., "?") or a Word object.

        Returns:
            The matching words.
        """
        query = str(word.value) if hasattr(word, "value") else str(word)
        inner_matches = self._inner.find_matches_str(query)
        return MatchWordList(
            inner_matches.word_length,
            inner=inner_matches,
        )

    def find_matches_str(self, query: str) -> MatchWordList:
        """Finds matches from a raw query string."""
        return self.find_matches(query)

    def sample(self, count: int = 1, temperature: float = 0.0) -> list[str]:
        """Samples a number of words from the list, with replacement.

        Args:
            count: The number of words to sample. Defaults to 1.
            temperature: Higher temperature will increase diversity of choices, and
                lower the impact of word scores. A temperature of zero will sample
                deterministically using score as the weight.

        Returns:
            The sampled words.
        """
        if len(self) == 0:
            return []
        scores = np.array(self._scores)
        if temperature > 0:
            scores = scores ** (1.0 / (temperature + 1e-6))
        weights = scores / (scores.sum() + 1e-12)
        indices = np.random.choice(len(self), size=count, replace=True, p=weights)
        return [self._words[i] for i in indices]

    @cached_property
    def frame(self) -> pd.DataFrame:
        """A pandas DataFrame representing the word list."""
        return pd.DataFrame({"word": self.words, "score": self.scores})

    def score_filter(self, threshold: float) -> WordList:
        """Returns a new word list containing only words with scores above the
        threshold.

        Args:
            threshold: The score threshold.

        Returns:
            The resulting WordList.
        """
        return WordList(self._inner.score_filter(threshold))

    def filter(self, filter_fn: Callable[[ScoredWord], bool]) -> WordList:
        """Returns a new word list containing only the words that pass the filter
        function.

        Args:
            filter_fn: A function that takes a word and its score, and returns True if
                the word should be kept.

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
            file,
            allow_pickle=True,
            words=self._words,
            scores=self._scores,
            **by_length_arrays,
        )

    def __len__(self):
        return len(self._inner)

    def __repr__(self):
        return f"WordList(\n{repr(self.frame)}\n)"

    def _repr_html_(self):
        return self.frame._repr_html_()

    def __getitem__(self, key: int) -> ScoredWord:
        if not isinstance(key, int):
            raise IndexError
        w, s = self._inner[key]
        return ScoredWord(w, s)

    def __contains__(self, item: str) -> bool:
        return self._inner.contains(item)

    def __eq__(self, other):
        if not isinstance(other, WordList):
            return False
        return self.words == other.words and self.scores == other.scores

    def __add__(self, other: WordList) -> WordList:
        if not isinstance(other, WordList):
            return NotImplemented
        return WordList(self._inner + other._inner)


class MatchWordList:
    """An object representing a filtered and scored WordList. Should not be
    constructed by the user, it is generated by matching methods on the original word
    list.
    """

    _inner: PyMatchWordList

    def __init__(
        self,
        word_length: int,
        words: list[str] | np.ndarray | None = None,
        scores: list[float] | np.ndarray | None = None,
        inner: PyMatchWordList | None = None,
    ):
        self._word_length = word_length
        self._words_cache: np.ndarray | None = None
        self._scores_cache: np.ndarray | None = None
        self._by_length_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if inner is not None:
            self._inner = inner
        else:
            w_list = [str(w) for w in words] if words is not None else []
            s_list = [float(s) for s in scores] if scores is not None else []
            self._inner = PyMatchWordList(word_length, w_list, s_list)

    @property
    def _words(self) -> np.ndarray:
        if self._words_cache is None:
            self._words_cache = np.asarray(self._inner.words, dtype=str)
        return self._words_cache

    @property
    def _scores(self) -> np.ndarray:
        if self._scores_cache is None:
            self._scores_cache = np.asarray(self._inner.scores, dtype=float)
        return self._scores_cache

    @property
    def _word_scores_by_length(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return {self._word_length: (self._words, self._scores)}

    @property
    def words(self) -> list[str]:
        return self._inner.words

    @property
    def scores(self) -> list[float]:
        return self._inner.scores

    def letter_scores_at_index(self, index: int) -> dict[str, float]:
        """The summed scores of matching letters at a given index.

        Args:
            index: The index to look at.

        Returns:
            A dict mapping letters to the summed scores of words containing that letter
            at the input index.
        """
        if len(self) > 0:
            return self._inner.letter_scores_at_index(index)
        else:
            return {}

    def rescore(
        self,
        rescore_fn: Callable[[str, float], float],
        drop_zeros: bool = True,
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
        rescored = self._inner.rescore(rescore_fn, drop_zeros)
        return MatchWordList(
            self._word_length,
            inner=rescored,
        )

    def score_filter(self, threshold: float) -> MatchWordList:
        """Returns a new word list containing only the words above the threshold.

        Args:
            threshold: The score threshold.

        Returns:
            The resulting MatchWordList
        """
        filtered = self._inner.score_filter(threshold)
        return MatchWordList(
            self._word_length,
            inner=filtered,
        )

    def filter_words(self, words: list[str]) -> MatchWordList:
        """Returns a new word list with a specific set of words filtered out.

        Args:
            words: The list of words to filter out.

        Returns:
            The new MatchWordlist.
        """
        filtered = self._inner.filter_words(words)
        return MatchWordList(
            self._word_length,
            inner=filtered,
        )

    def get_score(self, word: str) -> float | None:
        return self._inner.get_score(word)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, key: int) -> ScoredWord:
        if not isinstance(key, int):
            raise IndexError
        w, s = self._inner[key]
        return ScoredWord(w, s)

    def __contains__(self, item: str) -> bool:
        return item in self._inner


def _normalize(word: str) -> str:
    """Sanitizes an input word.

    Args:
        word: The input word.

    Returns:
        An upper-cased trimmed string.
    """
    return word.upper().replace(" ", "")


DEFAULT_WORDLIST = WordList()
