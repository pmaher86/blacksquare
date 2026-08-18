from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

from blacksquare.types import Direction, WordIndex
from blacksquare.word_list import INVERSE_CHARACTER_FREQUENCIES, MatchWordList, WordList

if TYPE_CHECKING:
    from blacksquare.cell import Cell
    from blacksquare.crossword import Crossword


class Word:
    """An object representing a single Word, with awareness of the parent grid. Should
    not be constructed by the user.
    """

    _parent: Crossword | None

    def __init__(
        self,
        parent_crossword: Crossword | None,
        direction: Direction,
        number: int,
        clue: str = "",
    ):
        self._parent = parent_crossword
        self._direction = direction
        self._number = number
        if (
            clue
            and parent_crossword is not None
            and hasattr(parent_crossword, "_inner")
        ):
            rust_dir = parent_crossword._to_rust_dir(direction)
            parent_crossword._inner.set_clue(rust_dir, number, clue)

    def __getitem__(self, key) -> Cell:
        return self.cells[key]

    def __setitem__(self, key, value):
        self.cells[key].value = value

    @property
    def direction(self) -> Direction:
        """The Direction of the word."""
        return self._direction

    @property
    def number(self) -> int:
        """The index number of the word."""
        return self._number

    @property
    def index(self) -> WordIndex:
        """The index (Direction, number) of the word."""
        return (self.direction, self.number)

    @property
    def clue(self) -> str:
        """The clue for the word."""
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            return self._parent._inner.get_clue(rust_dir, self.number)
        return ""

    @clue.setter
    def clue(self, value: str):
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            self._parent._inner.set_clue(rust_dir, self.number, str(value))

    @property
    def value(self) -> str:
        """The current string value of the word."""
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            val = self._parent._inner.get_word_value(rust_dir, self.number)
            if val is not None:
                return val
        return "".join([c.str for c in self.cells])

    @value.setter
    def value(self, value: str):
        if self._parent is not None:
            self._parent[self.index] = value
        else:
            for cell, v in zip(self.cells, value):
                cell.value = v

    @property
    def cells(self) -> list[Cell]:
        """A list of the cells that make up the word."""
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            cell_indices = self._parent._inner.get_word_cell_indices(
                rust_dir, self.number
            )
            if cell_indices is not None:
                return [self._parent[r, c] for r, c in cell_indices]
        return []

    def is_open(self) -> bool:
        """Returns True if the word contains open cells, False otherwise."""
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            return self._parent._inner.is_word_open(rust_dir, self.number)
        return any(c.is_open() for c in self.cells)

    @property
    def crosses(self) -> list[Word | None]:
        """Returns the words that cross the current word.

        Returns:
            A list of Word objects corresponding to the crosses.
        """
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            cross_tuples = self._parent._inner.get_word_crosses(rust_dir, self.number)
            if cross_tuples is not None:
                res: list[Word | None] = []
                for ct in cross_tuples:
                    if ct is not None:
                        py_dir = self._parent._from_rust_dir(ct[0])
                        res.append(self._parent[py_dir, ct[1]])
                    else:
                        res.append(None)
                return res
        return [cell.get_parent_word(self.direction.opposite) for cell in self.cells]

    @property
    def symmetric_image(self) -> Word | list[Word] | None:
        if self._parent is not None:
            res = self._parent.get_symmetric_word_index(self.index)
            if not res:
                return None
            elif isinstance(res, list):
                return [self._parent[i] for i in res]
            else:
                return self._parent[res]
        return None

    def find_matches(
        self, word_list: WordList | None = None, allow_repeats: bool = False
    ) -> MatchWordList:
        """Finds matches for the word, ranking matches by how many valid crosses they
        allow.

        Args:
            word_list: The word list to use for matching. If None, the default wordlist
                of the parent crossword is used.
            allow_repeats: Whether to include words that are already in the grid.
                Defaults to False.

        Returns:
            The matching words, scored by compatible crosses.
        """
        if self._parent is not None:
            effective_wl = self._parent.word_list if word_list is None else word_list
        else:
            effective_wl = word_list
        if effective_wl is None:
            raise ValueError("No word list provided")

        self_len = len(self)
        open_indices = [i for i, c in enumerate(self.cells) if c.is_open()]

        if hasattr(effective_wl, "_inner"):
            open_pos_list = []
            letter_weights_list = []
            for idx in open_indices:
                cross = self.crosses[idx]
                if cross is None:
                    continue
                cross_index = cross.crosses.index(self)
                cross_matches = effective_wl.find_matches(cross)
                scores_dict = cross_matches.letter_scores_at_index(cross_index)
                weights = [
                    scores_dict.get(chr(ord("A") + i), 0.0)
                    * INVERSE_CHARACTER_FREQUENCIES.get(chr(ord("A") + i), 1.0)
                    for i in range(26)
                ]
                open_pos_list.append(int(idx))
                letter_weights_list.append(weights)

            inner_res = effective_wl._inner.fused_cross_matches(
                self.value, open_pos_list, letter_weights_list
            )
            matches = MatchWordList(
                inner_res.word_length,
                inner=inner_res,
            )
            if not allow_repeats and self._parent is not None:
                matches = matches.filter_words(
                    [
                        w.value
                        for w in self._parent.iterwords()
                        if len(w) == self_len and not w.is_open()
                    ]
                )
            return matches

        matches = effective_wl.find_matches(self)
        if not allow_repeats and self._parent is not None:
            matches = matches.filter_words(
                [
                    w.value
                    for w in self._parent.iterwords()
                    if len(w) == self_len and not w.is_open()
                ]
            )
        letter_scores_per_index = {}
        for idx in open_indices:
            cross = self.crosses[idx]
            if cross is None:
                continue
            cross_index = cross.crosses.index(self)
            cross_matches = effective_wl.find_matches(cross)
            letter_scores_per_index[idx] = cross_matches.letter_scores_at_index(
                cross_index
            )

        def score_word_fn(word: str, score: float) -> float:
            per_letter_scores = [
                letter_scores_per_index[i].get(word[i], 0)
                * INVERSE_CHARACTER_FREQUENCIES.get(word[i], 1)
                for i in open_indices
                if i in letter_scores_per_index
            ]
            return math.log(math.prod(per_letter_scores) + 1.0) * score

        return matches.rescore(score_word_fn)

    def __repr__(self):
        return f'Word({self.direction.value} {self.number}: "{self.value.replace(" ", "?")}")'

    def __len__(self):
        if self._parent is not None and hasattr(self._parent, "_inner"):
            rust_dir = self._parent._to_rust_dir(self.direction)
            length = self._parent._inner.get_word_length(rust_dir, self.number)
            if length is not None:
                return length
        return len(self.cells)

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._parent = None
        return copied
