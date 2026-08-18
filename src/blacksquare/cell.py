from __future__ import annotations

import builtins
import copy
from typing import TYPE_CHECKING

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
        parent_crossword: Crossword | None,
        index: CellIndex,
        value: CellValue = SpecialCellValue.EMPTY,
    ):
        self._parent = parent_crossword
        self._index = index
        self._standalone_value = _parse_cell_input(value)
        self._standalone_shaded = False
        self._standalone_circled = False

    @property
    def parent_crossword(self) -> Crossword | None:
        return self._parent

    def get_parent_word(self, direction: Direction) -> Word | None:
        """Get the word to which the cell belongs in the given direction.

        Args:
            direction: The direction.

        Returns:
            The parent word.
        """
        if self._parent is not None:
            return self._parent.get_word_at_index(self.index, direction)
        return None

    def is_open(self) -> bool:
        return self.value == SpecialCellValue.EMPTY

    @property
    def value(self) -> CellValue:
        if self._parent is not None and hasattr(self._parent, "_inner"):
            val_str = self._parent._inner.get_cell_value(self._index[0], self._index[1])
            if val_str == " ":
                return SpecialCellValue.EMPTY
            elif val_str == "█":
                return SpecialCellValue.BLACK
            else:
                return val_str
        return self._standalone_value

    @value.setter
    def value(self, new_value):
        parsed = _parse_cell_input(new_value)
        if self._parent is not None:
            self._parent.set_cell(self._index, parsed)
        else:
            self._standalone_value = parsed

    @property
    def shaded(self) -> bool:
        if self._parent is not None and hasattr(self._parent, "_inner"):
            return self._parent._inner.get_cell_shaded(self._index[0], self._index[1])
        return self._standalone_shaded

    @shaded.setter
    def shaded(self, val: bool):
        if self._parent is not None and hasattr(self._parent, "_inner"):
            self._parent._inner.set_cell_shaded(self._index[0], self._index[1], val)
        else:
            self._standalone_shaded = val

    @property
    def circled(self) -> bool:
        if self._parent is not None and hasattr(self._parent, "_inner"):
            return self._parent._inner.get_cell_circled(self._index[0], self._index[1])
        return self._standalone_circled

    @circled.setter
    def circled(self, val: bool):
        if self._parent is not None and hasattr(self._parent, "_inner"):
            self._parent._inner.set_cell_circled(self._index[0], self._index[1], val)
        else:
            self._standalone_circled = val

    @property
    def index(self) -> CellIndex:
        return self._index

    @property
    def number(self) -> int | None:
        if self._parent is not None:
            return self._parent.get_cell_number(self._index)
        return None

    @property
    def symmetric_image(self) -> Cell | list[Cell] | None:
        if self._parent is not None:
            res = self._parent.get_symmetric_cell_index(self.index)
            if not res:
                return None
            elif isinstance(res, list):
                return [self._parent[i] for i in res]
            else:
                return self._parent[res]
        return None

    @property
    def str(self) -> builtins.str:
        val = self.value
        if isinstance(val, str):
            return val
        elif isinstance(val, SpecialCellValue):
            return val.str
        return str(val)

    def __repr__(self):
        return f"Cell({repr(self.value)})"

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.value == other.value
        else:
            return self.value == other

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._parent = None
        return copied


def _parse_cell_input(value: CellValue) -> CellValue:
    """Helper function to sanitize cell inputs.

    Args:
        value: The input value.

    Raises:
        ValueError: For invalid cell values.

    Returns:
        The cell value, either as a normalized string, or a SpecialCellValue enum.
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
