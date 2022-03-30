from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union, Dict, List, TYPE_CHECKING
import numpy as np
import pandas as pd
import re
from collections import defaultdict

from blacksquare.utils import sum_by_group
from blacksquare.types import SpecialCellValue


if TYPE_CHECKING:
    from blacksquare.word import Word


_ALPHA_REGEX = re.compile("^[A-Z]*$")


class WordList:
    def __init__(
        self,
        source: Union[
            str,
            Path,
            List[str],
            Dict[str, Union[int, float]],
        ],
    ):
        """Representation of a scored word list.

        Args:
            source (Union[ str, Path, List[str], Dict[str, Union[int, float]], ]): The
                source for the word list. Can be a list of strings, a dict of strings
                to scores, or a path to a .dict file with words in "word;score" format.
                Words will be normalized and scores will be scaled from 0-1.

        Raises:
            ValueError: If input type is not recognized
        """
        if isinstance(source, str) or isinstance(source, Path):
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
        norm_words, scores = unzipped[0], np.array(unzipped[1])
        norm_scores = scores / scores.max()

        self._words, self._scores = norm_words, norm_scores

        word_scores_by_length = defaultdict(lambda: ([], []))
        for w, s in zip(norm_words, norm_scores):
            word_scores_by_length[len(w)][0].append(w)
            word_scores_by_length[len(w)][1].append(s)

        self._word_scores_by_length = {
            l: (np.array(words, dtype=str), np.array(scores))
            for l, (words, scores) in word_scores_by_length.items()
        }

    def find_matches(self, word: Word) -> MatchWordList:
        """Find matches for a Word object.

        Args:
            word (Word): The word to match.

        Returns:
            MatchWordList: The matching words as a MatchWordList.
        """
        return self.find_matches_str(word.value)

    # TODO: lru cache?
    def find_matches_str(self, query: str) -> MatchWordList:
        """Find matches for a query string. Open letters can be represented by a " ",
        "?", or "_" character.

        Args:
            query (str): The string to match against (e.g. "M???ING")

        Returns:
            MatchWordList: A MatchWordList object containing the matching words.
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

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def scores(self) -> List[float]:
        return list(self._scores)

    def get_score(self, word: str) -> Optional[float]:
        """Return the score for a word.

        Args:
            word (str): The word to get the score for.

        Returns:
            Optional[float]: The score. None if word is not in word list.
        """
        words, scores = self._word_scores_by_length[len(word)]
        score = scores[np.where(words == word)]
        if len(score) == 1:
            return score[0]
        else:
            return None

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"word": self._words, "score": self._scores})

    def __len__(self):
        return len(self._words)

    def __repr__(self):
        return f"WordList(\n{repr(self.frame)}\n)"

    def _repr_html_(self):
        return self.frame._repr_html_()


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

    def letter_scores_at_index(self, index: int) -> Dict[str, int]:
        """The summed scores of matching letters at a given index.

        Args:
            index (int): The index to look at.

        Returns:
            Dict[str, int]: A dict mapping letters to the summed scores of words
            containing that letter at the input index
        """
        if len(self) > 0:
            letters = self._words.view("U1").reshape(self._words.size, -1)[:, index]
            return sum_by_group(letters, self._scores)
        else:
            return {}

    def rescore(
        self, rescore_fn: Callable[[str], float], drop_zeros=True
    ) -> MatchWordList:
        """Generates a new word list with new scores, as defined by the rescore
        function.

        Args:
            rescore_fn (Callable[[str], float]): The function mapping words to new
                scores. This function should treat zero as invalid.
            drop_zeros (bool, optional): Whether to remove words with a score of zero.
                Defaults to True.

        Returns:
            MatchWordList: A match word list where new scores are the results of the
                rescore function times the original score for the word.
        """
        vectorized_fn = np.vectorize(rescore_fn, otypes=[float])
        new_weights = vectorized_fn(self._words)
        new_scores = self._scores * new_weights
        new_words = self._words
        if drop_zeros:
            gt_zero_mask = new_scores > 0
            new_scores = new_scores[gt_zero_mask]
            new_words = new_words[gt_zero_mask]

        sort_indices = np.flip(np.argsort(new_scores))
        return MatchWordList(
            self._word_length, new_words[sort_indices], new_scores[sort_indices]
        )


def _normalize(word: str) -> str:
    """Sanitizes an input word.

    Args:
        word (str): The input word.

    Returns:
        str: An upper-cased trimmed string.
    """
    return word.upper().replace(" ", "")


DEFAULT_WORDLIST = WordList(Path(__file__).parent / "xwordlist.dict")
