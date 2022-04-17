from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, Union

from blacksquare.types import CellIndex, CellValue, Direction, SpecialCellValue

if TYPE_CHECKING:
    from blacksquare.crossword import Crossword
    from blacksquare.word import Word


class Cell:
    """An object representing a single cell in the crossword. Should not be constructed
    by the user.
    """

    def __init__(
        self,
        parent_crossword: Crossword,
        index: CellIndex,
        value: CellValue = SpecialCellValue.EMPTY,
    ):
        self._parent = parent_crossword
        self._index = index
        self._value = _parse_cell_input(value)
        self.shaded = False
        self.circled = False

    @property
    def parent_crossword(self) -> Crossword:
        return self._parent

    def get_parent_word(self, direction: Direction) -> Word:
        """Get the word to which the cell belongs in the given direction.

        Args:
            direction (Direction): The direction.

        Returns:
            Word: The parent word.
        """
        return self._parent.get_word_at_index(self.index, direction)

    @property
    def value(self) -> CellValue:
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = _parse_cell_input(new_value)

    @property
    def index(self) -> CellIndex:
        return self._index

    @property
    def number(self) -> Optional[int]:
        return self._parent.get_cell_number(self._index)

    @property
    def symmetric_image(self) -> Optional[Union[Cell, List[Cell]]]:
        result = self.parent_crossword.get_symmetric_cell_index(self.index)
        if not result:
            return
        elif isinstance(result, list):
            return [self.parent_crossword[i] for i in result]
        else:
            return self.parent_crossword[result]

    @property
    def str(self) -> str:
        if isinstance(self._value, str):
            return self._value
        elif isinstance(self._value, SpecialCellValue):
            return self._value.str

    def __repr__(self):
        return f"Cell({repr(self._value)})"

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self._value == other._value
        else:
            return self._value == other

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._parent = None
        return copied


def _parse_cell_input(value: CellValue) -> CellValue:
    """Helper function to sanitize cell inputs.

    Args:
        value (CellValue): The input value.

    Raises:
        ValueError: For invalid cell values.

    Returns:
        CellValue: The cell value, either as a normalized string, or a SpecialCellValue
        enum.
    """
    if isinstance(value, SpecialCellValue):
        return value
    elif not isinstance(value, str) or len(value) != 1:
        raise ValueError
    else:
        if value in SpecialCellValue.BLACK.input_str_reprs:
            return SpecialCellValue.BLACK
        elif value in SpecialCellValue.EMPTY.input_str_reprs:
            return SpecialCellValue.EMPTY
        else:
            return value.upper()
