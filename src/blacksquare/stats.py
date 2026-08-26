from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """The result of validating a Crossword grid."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        if self.is_valid:
            msg = "Validation passed: Puzzle is valid."
            if self.warnings:
                msg += f" (Warnings: {len(self.warnings)})"
            return msg
        err_list = "\n  - ".join(self.errors)
        return f"Validation failed ({len(self.errors)} error{'s' if len(self.errors) > 1 else ''}):\n  - {err_list}"

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, errors={self.errors}, warnings={self.warnings})"


@dataclass
class CrosswordStats:
    """Statistics for a Crossword grid."""

    total_words: int
    across_words: int
    down_words: int
    black_squares: int
    total_cells: int
    open_cells: int
    word_length_counts: dict[int, int]
    letter_counts: dict[str, int]
    rebus_count: int = 0
    circled_count: int = 0
    shaded_count: int = 0
    filled_words: int = 0
    open_words: int = 0

    @property
    def black_square_pct(self) -> float:
        """Percentage of the grid occupied by black squares."""
        return (
            (self.black_squares / self.total_cells) * 100.0
            if self.total_cells > 0
            else 0.0
        )

    @property
    def open_cell_pct(self) -> float:
        """Percentage of the grid occupied by open (non-black) cells."""
        return (
            (self.open_cells / self.total_cells) * 100.0
            if self.total_cells > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts statistics into a dictionary."""
        return {
            "total_words": self.total_words,
            "across_words": self.across_words,
            "down_words": self.down_words,
            "filled_words": self.filled_words,
            "open_words": self.open_words,
            "black_squares": self.black_squares,
            "total_cells": self.total_cells,
            "open_cells": self.open_cells,
            "black_square_pct": round(self.black_square_pct, 2),
            "open_cell_pct": round(self.open_cell_pct, 2),
            "word_length_counts": self.word_length_counts,
            "letter_counts": self.letter_counts,
            "rebus_count": self.rebus_count,
            "circled_count": self.circled_count,
            "shaded_count": self.shaded_count,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __str__(self) -> str:
        lines = [
            "CrosswordStats:",
            f"  Total Words: {self.total_words} (Across: {self.across_words}, Down: {self.down_words})",
            f"  Filled / Open Words: {self.filled_words} / {self.open_words}",
            f"  Black Squares: {self.black_squares} / {self.total_cells} ({self.black_square_pct:.1f}%)",
            "  Word Length Counts:",
        ]
        for length, count in sorted(self.word_length_counts.items(), reverse=True):
            lines.append(f"    {length:2d}-letter: {count:2d}")

        if self.letter_counts:
            top_letters = ", ".join(
                f"{k}: {v}"
                for k, v in list(self.letter_counts.items())[
                    : min(10, len(self.letter_counts))
                ]
            )
            if len(self.letter_counts) > 10:
                top_letters += f", ... ({len(self.letter_counts)} distinct entries)"
            lines.append(f"  Letter Counts: {top_letters}")

        if self.rebus_count or self.circled_count or self.shaded_count:
            decorations = []
            if self.rebus_count:
                decorations.append(f"Rebus: {self.rebus_count}")
            if self.circled_count:
                decorations.append(f"Circled: {self.circled_count}")
            if self.shaded_count:
                decorations.append(f"Shaded: {self.shaded_count}")
            lines.append(f"  Decorations: {', '.join(decorations)}")

        return "\n".join(lines)
