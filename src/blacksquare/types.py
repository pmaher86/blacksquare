from __future__ import annotations

import builtins
from enum import Enum


class Direction(Enum):
    """An Enum representing the directions of words in a crossword."""

    ACROSS = "Across"
    DOWN = "Down"

    @property
    def opposite(self) -> Direction:
        if self == Direction.ACROSS:
            return Direction.DOWN
        else:
            return Direction.ACROSS

    def __lt__(self, other) -> bool:
        if isinstance(other, Direction):
            return self == Direction.ACROSS and other == Direction.DOWN
        return NotImplemented

    def __repr__(self):
        return f"<{self.value}>"


class SpecialCellValue(Enum):
    "An enum representing blank and empty cell values in a crossword."

    BLACK = "Black"
    EMPTY = "Empty"

    @property
    def input_str_reprs(self) -> list[builtins.str]:
        if self == SpecialCellValue.BLACK:
            return [".", "#"]
        elif self == SpecialCellValue.EMPTY:
            return [" ", "-", "?", "_"]
        return []

    @property
    def str(self) -> builtins.str:
        if self == SpecialCellValue.BLACK:
            return "█"
        elif self == SpecialCellValue.EMPTY:
            return " "
        return ""

    def __repr__(self):
        return f"<{self.value}>"


ACROSS = Direction.ACROSS
DOWN = Direction.DOWN

WordIndex = tuple[Direction, int]
CellIndex = tuple[int, int]
CellValue = builtins.str | SpecialCellValue
