from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from blacksquare.types import Direction, SpecialCellValue, WordIndex
from blacksquare.word_list import (INVERSE_CHARACTER_FREQUENCIES,
                                   MatchWordList, WordList)

if TYPE_CHECKING:
    from blacksquare.cell import Cell
    from blacksquare.crossword import Crossword


class Word:
    """An object representing a single Word, with awareness of the parent grid. Should
    not be constructed by the user.
    """

    def __init__(
        self,
        parent_crossword: Crossword,
        direction: Direction,
        number: int,
        clue: str = "",
    ):
        """Returns a new Word object.

        Args:
            parent_crossword (Crossword): The parent crossword for the word.
            direction (Direction): The direction of the word.
            number (int): The ordinal number of the word.
            clue (str, optional): The clue associated with the word. Defaults to "".
        """
        self.clue = clue
        self._parent = parent_crossword
        self._direction = direction
        self._number = number

    def __getitem__(self, key) -> Cell:
        return self.cells[key]

    def __setitem__(self, key, value):
        self.cells[key].value = value
        # grid_indices = self._parent.get_indices(self.index)
        # self._parent[grid_indices[key]] = value

    @property
    def direction(self) -> Direction:
        """Direction: The direction of the word."""
        return self._direction

    @property
    def number(self) -> int:
        """int: The number of the word."""
        return self._number

    @property
    def index(self) -> WordIndex:
        """Tuple[Direction, int]: The (direction, number) index of the word."""
        return (self.direction, self.number)

    @property
    def cells(self) -> List[Cell]:
        return self._parent.get_word_cells(self.index)

    @property
    def value(self) -> str:
        # TODO: rename to str
        """str: The current fill value of the word"""
        # return "".join(self._parent.grid[self._parent._get_word_mask(self.index)])
        return "".join([c.str for c in self.cells])

    # Todo: array, str?

    @value.setter
    def value(self, new_value: str):
        self._parent[self.index] = new_value

    def is_open(self) -> bool:
        """Does the word contain any blank spaces.

        Returns:
            bool: True if any of the letters are blank.
        """
        return np.equal(self.cells, SpecialCellValue.EMPTY).any()

    @property
    def crosses(self) -> List[Word]:
        """Returns the words that cross the current word.

        Returns:
            list[Word]: A list of Word objects corresponding to the crosses.
        """
        return [cell.get_parent_word(self.direction.opposite) for cell in self.cells]

    @property
    def symmetric_image(self) -> Optional[Union[Word, List[Word]]]:
        result = self._parent.get_symmetric_word_index(self.index)
        if not result:
            return
        elif isinstance(result, list):
            return [self._parent[i] for i in result]
        else:
            return self._parent[result]

    def find_matches(self, word_list: Optional[WordList] = None) -> MatchWordList:
        """Finds matches for the word, ranking matches by how many valid crosses they
        allow.

        Args:
            word_list (Optional[WordList], optional): The word list to use for matching.
            If None, the default wordlist of the parent crossword is used..

        Returns:
            MatchWordList: The matching words, scored by compatible crosses.
        """
        word_list = self._parent.word_list if word_list is None else word_list
        matches = word_list.find_matches(self)
        open_indices = np.argwhere(
            np.equal(self.cells, SpecialCellValue.EMPTY)
        ).squeeze(axis=1)
        letter_scores_per_index = {}
        for idx in open_indices:
            cross = self.crosses[idx]
            if cross is None:
                continue
            cross_index = cross.crosses.index(self)
            cross_matches = word_list.find_matches(cross)
            letter_scores_per_index[idx] = cross_matches.letter_scores_at_index(
                cross_index
            )

        def score_word_fn(word: str, score: float) -> float:
            per_letter_scores = [
                letter_scores_per_index[i].get(word[i], 0)
                * INVERSE_CHARACTER_FREQUENCIES.get(word[i], 1)
                for i in open_indices if i in letter_scores_per_index
            ]
            return np.log(np.prod(per_letter_scores) + 1) * score

        return matches.rescore(score_word_fn)

    def __repr__(self):
        return f'Word({self.direction.value} {self.number}: "{self.value.replace(" ", "?")})"'

    def __len__(self):
        return len(self.cells)

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._parent = None
        return copied
