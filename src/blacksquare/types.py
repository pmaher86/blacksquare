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


class Rebus:
    """Represents a rebus cell in a crossword with across and down text values."""

    _across: builtins.str
    _down: builtins.str

    def __init__(
        self,
        value: builtins.str | None = None,
        down: builtins.str | None = None,
        *,
        across: builtins.str | None = None,
    ) -> None:
        if value is not None and across is not None:
            raise ValueError(
                "Cannot specify both positional value and keyword 'across'"
            )

        if (
            value is not None
            and isinstance(value, str)
            and "/" in value
            and down is None
            and across is None
        ):
            parts = value.split("/")
            if len(parts) == 2:
                act_across = parts[0]
                act_down = parts[1]
            else:
                act_across = value
                act_down = value
        else:
            act_across = across if across is not None else value
            act_down = down

        if act_across is None and act_down is None:
            raise ValueError("Must specify at least one rebus value")
        elif act_across is not None and act_down is None:
            act_down = act_across
        elif act_across is None and act_down is not None:
            act_across = act_down

        assert act_across is not None and act_down is not None

        if not isinstance(act_across, str) or not isinstance(act_down, str):
            raise ValueError("Rebus values must be strings")

        clean_across = act_across.strip().upper()
        clean_down = act_down.strip().upper()

        if not clean_across or not clean_down:
            raise ValueError("Rebus values cannot be empty")

        self._across = clean_across
        self._down = clean_down

    @property
    def across(self) -> builtins.str:
        """The across string value of the rebus."""
        return self._across

    @property
    def down(self) -> builtins.str:
        """The down string value of the rebus."""
        return self._down

    @property
    def value(self) -> builtins.str | tuple[builtins.str, builtins.str]:
        """The value of the rebus. Returns str if symmetric, else (across, down) tuple."""
        if self.is_symmetric:
            return self._across
        return (self._across, self._down)

    @property
    def is_symmetric(self) -> bool:
        """Whether the across and down values are identical."""
        return self._across == self._down

    def get_value(self, direction: Direction) -> builtins.str:
        """Returns the string value for the given direction."""
        if direction == Direction.ACROSS:
            return self._across
        elif direction == Direction.DOWN:
            return self._down
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def __str__(self) -> builtins.str:
        if self.is_symmetric:
            return self._across
        return f"{self._across}/{self._down}"

    def __repr__(self) -> builtins.str:
        if self.is_symmetric:
            return f"Rebus({repr(self._across)})"
        return f"Rebus(across={repr(self._across)}, down={repr(self._down)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rebus):
            return self._across == other._across and self._down == other._down
        elif isinstance(other, str):
            clean = other.strip().upper()
            return self._across == clean and self._down == clean
        return False

    def __hash__(self) -> int:
        return hash((self._across, self._down))


ACROSS = Direction.ACROSS
DOWN = Direction.DOWN

WordIndex = tuple[Direction, int]
CellIndex = tuple[int, int]
CellValue = builtins.str | SpecialCellValue | Rebus
