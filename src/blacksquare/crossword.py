from __future__ import annotations

import importlib
import io
import os
from collections.abc import Iterable, Iterator
from secrets import token_hex
from typing import Any, BinaryIO, overload

import numpy as np
import rich.box
from rich.console import Console
from rich.table import Table

from blacksquare._blacksquare_rs import (
    Direction as RustDirection,
)
from blacksquare._blacksquare_rs import (
    PyCrossword,
)
from blacksquare._blacksquare_rs import (
    Symmetry as RustSymmetry,
)
from blacksquare.cell import Cell
from blacksquare.html import CSS_TEMPLATE
from blacksquare.puz import (
    BLACKSQUARE,
    BLACKSQUARE2,
    BLANKSQUARE,
    ExtensionCode,
    GridMarkup,
    PuzData,
    parse_rebus_table,
    serialize_rebus_table,
)
from blacksquare.stats import CrosswordStats, ValidationResult
from blacksquare.symmetry import Symmetry
from blacksquare.types import (
    CellIndex,
    CellValue,
    Direction,
    Rebus,
    SpecialCellValue,
    WordIndex,
)
from blacksquare.utils import is_intlike
from blacksquare.word import Word
from blacksquare.word_list import DEFAULT_WORDLIST, WordList

weasyprint: Any = None
pypdf: Any = None
try:
    pypdf = importlib.import_module("pypdf")
    weasyprint = importlib.import_module("weasyprint")
except ImportError:
    pass

BLACK, EMPTY = SpecialCellValue.BLACK, SpecialCellValue.EMPTY
ACROSS, DOWN = Direction.ACROSS, Direction.DOWN


class Crossword:
    """An object representing a crossword puzzle backed by an ultra-fast Rust engine."""

    def __init__(
        self,
        num_rows: int | None = None,
        num_cols: int | None = None,
        grid: list[list[str]] | np.ndarray | None = None,
        symmetry: Symmetry | None = Symmetry.ROTATIONAL,
        word_list: WordList | None = None,
        display_size_px: int = 450,
        _inner: PyCrossword | None = None,
    ):
        """Creates a new Crossword object."""
        self._cached_cells: dict[CellIndex, Cell] = {}
        self._cached_words: dict[WordIndex, Word] = {}

        if _inner is not None:
            self._inner = _inner
        else:
            assert (num_rows is not None) ^ (grid is not None), (
                "Either specify shape or provide grid."
            )

            rust_sym = self._to_rust_sym(symmetry)

            if num_rows:
                n_rows = num_rows
                n_cols = num_cols if num_cols else n_rows
                if symmetry and symmetry.requires_square and n_rows != n_cols:
                    raise ValueError(
                        f"{symmetry.value} symmetry requires a square grid."
                    )
                self._inner = PyCrossword(
                    num_rows=n_rows,
                    num_cols=n_cols,
                    grid=None,
                    symmetry=rust_sym,
                    display_size_px=display_size_px,
                )
            elif grid is not None:
                assert np.all([len(r) == len(grid[0]) for r in grid])
                grid_list = []
                for r in grid:
                    row_strs = []
                    for val in r:
                        if isinstance(val, Cell):
                            row_strs.append(val.str)
                        elif isinstance(val, SpecialCellValue):
                            row_strs.append(val.str)
                        else:
                            row_strs.append(str(val))
                    grid_list.append(row_strs)

                n_rows = len(grid_list)
                n_cols = len(grid_list[0])
                if symmetry and symmetry.requires_square and n_rows != n_cols:
                    raise ValueError(
                        f"{symmetry.value} symmetry requires a square grid."
                    )

                self._inner = PyCrossword(
                    num_rows=None,
                    num_cols=None,
                    grid=grid_list,
                    symmetry=rust_sym,
                    display_size_px=display_size_px,
                )

        self.word_list = word_list if word_list is not None else DEFAULT_WORDLIST

    @staticmethod
    def _to_rust_sym(sym: Symmetry | None) -> RustSymmetry | None:
        if sym is None:
            return None
        mapping = {
            Symmetry.ROTATIONAL: RustSymmetry.Rotational,
            Symmetry.FULL: RustSymmetry.Full,
            Symmetry.VERTICAL: RustSymmetry.Vertical,
            Symmetry.HORIZONTAL: RustSymmetry.Horizontal,
            Symmetry.BIAXIAL: RustSymmetry.Biaxial,
            Symmetry.NE_DIAGONAL: RustSymmetry.NeDiagonal,
            Symmetry.NW_DIAGONAL: RustSymmetry.NwDiagonal,
        }
        return mapping.get(sym)

    @staticmethod
    def _from_rust_sym(sym: RustSymmetry | None) -> Symmetry | None:
        if sym is None:
            return None
        return Symmetry(sym.value)

    @staticmethod
    def _to_rust_dir(d: Direction) -> RustDirection:
        return RustDirection.Across if d == Direction.ACROSS else RustDirection.Down

    @staticmethod
    def _from_rust_dir(d: RustDirection) -> Direction:
        return Direction.ACROSS if d.value == "Across" else Direction.DOWN

    @property
    def num_rows(self) -> int:
        """The number of rows in the puzzle"""
        return self._inner.num_rows

    @property
    def num_cols(self) -> int:
        """The number of columns in the puzzle"""
        return self._inner.num_cols

    @property
    def symmetry(self) -> Symmetry | None:
        return self._from_rust_sym(self._inner.symmetry)

    @symmetry.setter
    def symmetry(self, sym: Symmetry | None):
        self._inner.symmetry = self._to_rust_sym(sym)

    @property
    def display_size_px(self) -> int:
        return self._inner.display_size_px

    @display_size_px.setter
    def display_size_px(self, px: int):
        self._inner.display_size_px = px

    @property
    def _grid(self) -> np.ndarray:
        rows, cols = self.num_rows, self.num_cols
        cells = [self[r, c] for r in range(rows) for c in range(cols)]
        return np.array(cells, dtype=object).reshape((rows, cols))

    @property
    def _numbers(self) -> np.ndarray:
        return np.array(self._inner.numbers_grid(), dtype=int)

    @property
    def _across(self) -> np.ndarray:
        return np.array(self._inner.across_numbers_grid(), dtype=int)

    @property
    def _down(self) -> np.ndarray:
        return np.array(self._inner.down_numbers_grid(), dtype=int)

    @property
    def _words(self) -> dict[WordIndex, Word]:
        res = {}
        for r_dir, num in self._inner.iter_word_indices():
            py_dir = self._from_rust_dir(r_dir)
            res[(py_dir, num)] = self[py_dir, num]
        return res

    @overload
    def __getitem__(self, key: tuple[Direction, int]) -> Word: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> Cell: ...
    @overload
    def __getitem__(self, key: WordIndex) -> Word: ...
    @overload
    def __getitem__(self, key: CellIndex) -> Cell: ...
    def __getitem__(self, key: CellIndex | WordIndex | tuple[Any, Any]) -> Cell | Word:
        if isinstance(key, tuple) and len(key) == 2:
            k0, k1 = key
            if isinstance(k0, Direction) and is_intlike(k1):
                num = int(k1)
                word_idx = (k0, num)
                r_dir = self._to_rust_dir(k0)
                val = self._inner.get_word_value(r_dir, num)
                if val is not None:
                    if word_idx not in self._cached_words:
                        self._cached_words[word_idx] = Word(self, k0, num)
                    return self._cached_words[word_idx]
                else:
                    raise IndexError
            elif not isinstance(k0, Direction) and is_intlike(k0) and is_intlike(k1):
                r, c = int(k0), int(k1)
                if r < 0:
                    r += self.num_rows
                if c < 0:
                    c += self.num_cols
                if 0 <= r < self.num_rows and 0 <= c < self.num_cols:
                    if (r, c) not in self._cached_cells:
                        self._cached_cells[(r, c)] = Cell(self, (r, c))
                    return self._cached_cells[(r, c)]
                else:
                    raise IndexError
        raise IndexError

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            k0, k1 = key
            if isinstance(k0, Direction) and is_intlike(k1):
                self.set_word((k0, int(k1)), value)
            elif not isinstance(k0, Direction) and is_intlike(k0) and is_intlike(k1):
                self.set_cell((int(k0), int(k1)), value)
            else:
                raise IndexError
        else:
            raise IndexError

    def set_cell(self, index: CellIndex, value: CellValue) -> None:
        """Sets a cell to a new value.

        Args:
            index: The index of the cell.
            value: The new value of the cell.
        """
        if isinstance(value, (list, tuple, int, float, Crossword, Word)):
            raise ValueError(f"Invalid cell value type: {type(value)}")

        r, c = int(index[0]), int(index[1])
        if r < 0:
            r += self.num_rows
        if c < 0:
            c += self.num_cols
        if not (0 <= r < self.num_rows and 0 <= c < self.num_cols):
            raise IndexError(f"Cell index {(r, c)} out of bounds")

        if isinstance(value, Rebus):
            self._inner.set_cell_rebus(r, c, value.across, value.down)
        elif isinstance(value, SpecialCellValue):
            self._inner.set_cell_value(r, c, value.str)
        elif isinstance(value, Cell):
            if isinstance(value.value, Rebus):
                self._inner.set_cell_rebus(r, c, value.value.across, value.value.down)
            else:
                self._inner.set_cell_value(r, c, value.str)
        elif isinstance(value, str):
            if value in SpecialCellValue.BLACK.input_str_reprs:
                self._inner.set_cell_value(r, c, SpecialCellValue.BLACK.str)
            elif value in SpecialCellValue.EMPTY.input_str_reprs:
                self._inner.set_cell_value(r, c, SpecialCellValue.EMPTY.str)
            elif len(value) == 1:
                self._inner.set_cell_value(r, c, value.upper())
            else:
                clean = value.strip()
                if clean in SpecialCellValue.BLACK.input_str_reprs:
                    self._inner.set_cell_value(r, c, SpecialCellValue.BLACK.str)
                elif clean in SpecialCellValue.EMPTY.input_str_reprs:
                    self._inner.set_cell_value(r, c, SpecialCellValue.EMPTY.str)
                elif len(clean) == 1:
                    self._inner.set_cell_value(r, c, clean.upper())
                else:
                    raise ValueError(f"Invalid cell value length: {value!r}")
        else:
            raise ValueError(f"Invalid cell value type: {type(value)}")

    def set_word(self, word_index: WordIndex, value: str) -> None:
        """Sets a word to a new value.

        Args:
            word_index: The index of the word.
            value: The new value of the word.
        """
        if not isinstance(value, str):
            raise ValueError(f"Word value must be str, got {type(value)}")

        direction, num = word_index[0], int(word_index[1])
        r_dir = self._to_rust_dir(direction)
        cell_indices = self._inner.get_word_cell_indices(r_dir, num)
        if cell_indices is None:
            raise IndexError(f"Word {word_index} not found in grid")

        word_cells = [self[r, c] for r, c in cell_indices]
        new_values = _parse_word_string_to_cell_values(word_cells, direction, value)
        for (r, c), new_val in zip(cell_indices, new_values):
            self.set_cell((r, c), new_val)

    def get_cell_number(self, cell_index: CellIndex) -> int | None:
        """Gets the crossword numeral at a given cell, if it exists.

        Args:
            cell_index: The index of the cell.

        Returns:
            The crossword number in that cell, if any.
        """
        r, c = int(cell_index[0]), int(cell_index[1])
        if r < 0:
            r += self.num_rows
        if c < 0:
            c += self.num_cols
        return self._inner.get_cell_number(r, c)

    def get_word_cells(self, word_index: WordIndex) -> list[Cell]:
        """Gets the cells for a word index.

        Args:
            word_index: The word index.

        Returns:
            The list of Cells in the word.
        """
        r_dir = self._to_rust_dir(word_index[0])
        num = int(word_index[1])
        coords = self._inner.get_word_cell_indices(r_dir, num)
        if coords is not None:
            return [self[r, c] for r, c in coords]
        return []

    def get_indices(self, word_index: WordIndex) -> list[CellIndex]:
        """Gets the list of cell indices for a given word.

        Args:
            word_index: The index of the desired word.

        Returns:
            A list of cell indices that belong to the word.
        """
        r_dir = self._to_rust_dir(word_index[0])
        num = int(word_index[1])
        coords = self._inner.get_word_cell_indices(r_dir, num)
        if coords is not None:
            return coords
        raise IndexError(f"Word {word_index} not found")

    def get_word_at_index(self, index: CellIndex, direction: Direction) -> Word | None:
        """Gets the word that passes through a cell in a given direction.

        Args:
            index: The index of the cell.
            direction: The direction of the word.

        Returns:
            The word passing through the index in the provided direction.
        """
        r, c = int(index[0]), int(index[1])
        if r < 0:
            r += self.num_rows
        if c < 0:
            c += self.num_cols
        r_dir = self._to_rust_dir(direction)
        res = self._inner.get_word_at_cell(r, c, r_dir)
        if res is not None:
            py_dir = self._from_rust_dir(res[0])
            return self[py_dir, res[1]]
        return None

    def get_symmetric_cell_index(
        self, index: CellIndex, force_list: bool = False
    ) -> CellIndex | list[CellIndex] | None:
        """Gets the index of a symmetric grid cell. Useful for enforcing symmetry.

        Args:
            index: The input cell index.
            force_list: Whether to require that single indices are returned as a list.

        Returns:
            The index (or indices) of the cell symmetric to the input.
        """
        if not self.symmetry:
            return [] if force_list else None
        r, c = int(index[0]), int(index[1])
        if r < 0:
            r += self.num_rows
        if c < 0:
            c += self.num_cols
        images = self._inner.get_symmetric_cell_indices(r, c)
        if not images:
            return [] if force_list else None
        if self.symmetry.is_multi_image or force_list:
            return images
        else:
            return images[0]

    def get_symmetric_word_index(
        self, word_index: WordIndex, force_list: bool = False
    ) -> WordIndex | list[WordIndex] | None:
        """Gets the index of a symmetric word. Useful for enforcing symmetry.

        Args:
            word_index: The input word index.
            force_list: Whether to require that single indices are returned as a list.

        Returns:
            The index (or indices) of the word symmetric to the input.
        """
        if not self.symmetry:
            return [] if force_list else None
        r_dir = self._to_rust_dir(word_index[0])
        images = self._inner.get_symmetric_word_indices(r_dir, int(word_index[1]))
        py_images = [(self._from_rust_dir(d), num) for d, num in images]
        if not py_images:
            return [] if force_list else None
        if self.symmetry.is_multi_image or force_list:
            return py_images
        else:
            return py_images[0]

    def get_disconnected_open_subgrids(self) -> list[list[WordIndex]]:
        """Returns a list of open subgrids, as represented by a list of words.

        Returns:
            A list of open subgrids.
        """
        raw_subs = self._inner.get_disconnected_open_subgrids()
        return [[(self._from_rust_dir(d), num) for d, num in sub] for sub in raw_subs]

    def hashable_state(
        self, word_indices: list[WordIndex]
    ) -> tuple[tuple[WordIndex, str], ...]:
        """Returns a list of tuple of (word index, current value) pairs in sorted order.

        Args:
            word_indices: The list of word indices of interest.

        Returns:
            A tuple of (word index, value) tuples
        """
        sorted_indices = sorted(word_indices)
        return tuple((i, self[i].value) for i in sorted_indices)

    def iterwords(
        self, direction: Direction | None = None, only_open: bool = False
    ) -> Iterator[Word]:
        """Method for iterating over the words in the crossword.

        Args:
            direction: If provided, limits the iterator to only the given direction.
            only_open: Whether to only return open words. Defaults to False.

        Yields:
            An iterator of Word objects.
        """
        r_dir = self._to_rust_dir(direction) if direction is not None else None
        for r_d, num in self._inner.iter_word_indices(r_dir, only_open):
            py_dir = self._from_rust_dir(r_d)
            yield self[py_dir, num]

    def itercells(self) -> Iterator[Cell]:
        """Method for iterating over the cells in the crossword.

        Yields:
            An iterator of Cell objects. Ordered left to right, top to bottom.
        """
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                yield self[r, c]

    @property
    def clues(self) -> dict[WordIndex, str]:
        """A dict mapping word index to clue."""
        return {
            (self._from_rust_dir(d), num): clue
            for (d, num), clue in self._inner.get_clues()
        }

    def copy(self) -> Crossword:
        """Returns a copy of the current crossword.

        Returns:
            A copy of the current Crossword object.
        """
        return Crossword(
            _inner=self._inner.copy(),
            word_list=self.word_list,
            display_size_px=self.display_size_px,
        )

    def __deepcopy__(self, memo):
        return self.copy()

    def __repr__(self):
        words = list(self.iterwords())
        if not words:
            return 'Crossword("")'
        longest_filled_word = max(words, key=lambda w: len(w) if not w.is_open() else 0)
        return f'Crossword("{longest_filled_word.value}")'

    def fill(
        self,
        word_list: WordList | None = None,
        timeout: float | None = 30.0,
        temperature: float = 0.0,
        score_filter: float | None = None,
        allow_repeats: bool = False,
        upweight_diverse_letters: bool = False,
        show_progress: bool = True,
    ) -> Crossword | None:
        """Searches for a possible fill, and returns the result as a new Crossword
        object. Backed by the native Rust backtracking solver.

        Args:
            word_list: An optional word list to use instead of the default.
            timeout: The maximum time in seconds to search before returning.
            temperature: A parameter to control randomness.
            score_filter: A threshold to apply to the word list before filling.
            allow_repeats: Whether to allow duplicate words in the grid.
            upweight_diverse_letters: Whether to upweight rare/diverse letters
                (J, Z, Q, X, etc.) during crossing candidate evaluation.
                Defaults to False.
            show_progress: Whether to display live in-progress grid updates
                in the terminal for long-running searches (>100ms). Defaults to True.

        Returns:
            The filled Crossword, or None if no solution found / timed out.
        """
        wl = word_list if word_list is not None else self.word_list
        filled_inner = self._inner.fill(
            wl._inner,
            timeout=timeout,
            temperature=temperature,
            score_filter=score_filter,
            allow_repeats=allow_repeats,
            upweight_diverse_letters=upweight_diverse_letters,
            show_progress=show_progress,
        )
        if filled_inner is not None:
            return Crossword(
                _inner=filled_inner,
                word_list=wl,
                display_size_px=self.display_size_px,
            )
        return None

    @classmethod
    def from_puz(
        cls,
        source: str | os.PathLike[str] | bytes | BinaryIO,
        word_list: WordList | None = None,
    ) -> Crossword:
        """Creates a Crossword object from a .puz file path, bytes, or file-like object.

        Restores the full grid, words, clues, rebus cells (from GRBS/RTBL extensions),
        and circled cells (from GEXT extension).
        """
        if isinstance(source, (str, os.PathLike)):
            with open(source, "rb") as f:
                data = f.read()
        elif isinstance(source, bytes):
            data = source
        elif hasattr(source, "read"):
            data = source.read()
        else:
            raise TypeError(f"Unsupported source type for from_puz: {type(source)}")

        puz_data = PuzData.from_bytes(data)
        xw = cls(
            num_rows=puz_data.height,
            num_cols=puz_data.width,
            symmetry=None,
            word_list=word_list,
        )

        # Rebus reconstruction
        rebus_dict: dict[int, str] = {}
        if ExtensionCode.RebusSolutions in puz_data.extensions:
            rtbl_str = puz_data.extensions[ExtensionCode.RebusSolutions].decode(
                puz_data.encoding, "replace"
            )
            rebus_dict = parse_rebus_table(rtbl_str)

        grbs_data = puz_data.extensions.get(ExtensionCode.Rebus, b"")
        gext_data = puz_data.extensions.get(ExtensionCode.Markup, b"")

        for r in range(puz_data.height):
            for c in range(puz_data.width):
                idx = r * puz_data.width + c
                ch = puz_data.solution[idx]
                if ch in [BLACKSQUARE, BLACKSQUARE2, "#"]:
                    xw[r, c] = SpecialCellValue.BLACK
                elif grbs_data and idx < len(grbs_data) and grbs_data[idx] > 0:
                    k = grbs_data[idx] - 1
                    rebus_val = rebus_dict.get(k, ch)
                    xw[r, c] = Rebus(rebus_val)
                elif ch in [BLANKSQUARE, " ", "?", "_"]:
                    xw[r, c] = SpecialCellValue.EMPTY
                else:
                    xw[r, c] = ch

                if gext_data and idx < len(gext_data):
                    if gext_data[idx] & GridMarkup.Circled:
                        xw[r, c].circled = True

        sorted_words = sorted(
            list(xw.iterwords()), key=lambda w: (w.number, w.direction)
        )
        for w, clue_text in zip(sorted_words, puz_data.clues):
            w.clue = clue_text

        return xw

    def to_puz(
        self,
        target: str | os.PathLike[str] | BinaryIO | None = None,
        *,
        title: str = "",
        author: str = "",
        copyright: str = "",
        notes: str = "",
    ) -> bytes:
        """Exports the Crossword object to Across Lite .puz binary format.

        Saves full grid solutions, clues, rebus cells (via GRBS and RTBL extensions),
        and circled cells (via GEXT extension). If target is provided, writes to the
        file or stream; otherwise returns the raw bytes.
        """
        puz_data = PuzData(version="1.3")
        puz_data.width = self.num_cols
        puz_data.height = self.num_rows
        puz_data.title = title
        puz_data.author = author
        puz_data.copyright = copyright
        puz_data.notes = notes

        n_cells = self.num_rows * self.num_cols
        sol_chars: list[str] = []
        fill_chars: list[str] = []

        rebus_map: dict[str, int] = {}
        grbs_bytes = bytearray(n_cells)
        gext_bytes = bytearray(n_cells)
        has_rebus = False
        has_gext = False

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                idx = r * self.num_cols + c
                cell = self[r, c]
                if cell == SpecialCellValue.BLACK:
                    sol_chars.append(BLACKSQUARE)
                    fill_chars.append(BLACKSQUARE)
                else:
                    fill_chars.append(BLANKSQUARE)
                    if isinstance(cell.value, Rebus):
                        has_rebus = True
                        rebus_str = str(cell.value)
                        if rebus_str not in rebus_map:
                            rebus_map[rebus_str] = len(rebus_map)
                        k = rebus_map[rebus_str]
                        grbs_bytes[idx] = k + 1
                        sol_chars.append(cell.value.across[0])
                    elif cell.is_open():
                        sol_chars.append(BLANKSQUARE)
                    else:
                        sol_chars.append(cell.str[0] if cell.str else BLANKSQUARE)

                if cell.circled:
                    has_gext = True
                    gext_bytes[idx] |= GridMarkup.Circled

        puz_data.solution = "".join(sol_chars)
        puz_data.fill = "".join(fill_chars)

        sorted_words = sorted(
            list(self.iterwords()), key=lambda w: (w.number, w.direction)
        )
        puz_data.clues = [w.clue or "" for w in sorted_words]

        if has_rebus:
            puz_data.extensions[ExtensionCode.Rebus] = bytes(grbs_bytes)
            inv_rebus = {k: v for v, k in rebus_map.items()}
            puz_data.extensions[ExtensionCode.RebusSolutions] = puz_data.encode(
                serialize_rebus_table(inv_rebus)
            )

        if has_gext:
            puz_data.extensions[ExtensionCode.Markup] = bytes(gext_bytes)

        raw_bytes = puz_data.to_bytes()

        if isinstance(target, (str, os.PathLike)):
            with open(target, "wb") as f:
                f.write(raw_bytes)
        elif hasattr(target, "write"):
            target.write(raw_bytes)

        return raw_bytes

    def to_pdf(
        self,
        filename: str,
        header: list[str] | None = None,
    ) -> None:
        """Outputs a .pdf file in NYT submission format from the Crossword object."""
        if weasyprint is None:
            raise ImportError(
                "Can't import weasyprint, run pip install blacksquare[pdf] to install."
            )

        header_html = "<br />".join(header) if header else ""
        grid_html = f"""
            <html>
            <head><meta charset="utf-8">
            <style>
            @page {{
                margin:0.25 in;
                margin-bottom: 0;
            }}

            @media print {{
            div {{
                break-inside: avoid-page !important;
            }}
            }}
            </style>
            </head>
            <body>
            <div style='font-size:14pt; break-after: avoid-page !important;'>
                {header_html}
            </div>
            <br /> <br /> <br /> <br />
            <div style='margin: auto;'>
                {self._grid_html(size_px=600)}
            </div>
            </body></html>
        """

        row_template = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>"

        def clue_rows(direction):
            row_strings = [
                row_template.format(w.number, w.clue, w.value)
                for w in self.iterwords(direction)
            ]
            return "".join(row_strings)

        clue_html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    td {{vertical-align:top;}}
                    table {{
                        text-align:left;
                        width:100%;
                        font-size:16pt;
                        border-spacing:1rem;
                    }}
                </style>
            </head>
            <body>
            <table><tbody>
            <tr><td colspan="3">ACROSS</td></tr>
            {clue_rows(ACROSS)}
            <tr><td></td></tr>
            <tr><td colspan="3">DOWN</td></tr>
            {clue_rows(DOWN)}
            </tbody></table>
            </body></html>
        """
        merger = pypdf.PdfWriter()
        for html_page in [grid_html, clue_html]:
            pdf = weasyprint.HTML(string=html_page, encoding="UTF-8").write_pdf()
            merger.append(pypdf.PdfReader(io.BytesIO(pdf)))
        merger.write(str(filename))
        merger.close()

    def _text_grid(self, numbers: bool = False) -> Table:
        """Returns a rich Table that displays the crossword."""
        superscripts = ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")
        table = Table(
            box=rich.box.SQUARE,
            show_header=False,
            show_lines=True,
            width=4 * self.num_cols + 1,
            padding=0,
        )
        for c in range(self.num_cols):
            table.add_column(justify="left", width=3)
        for r in range(self.num_rows):
            strings = []
            for c in range(self.num_cols):
                cell = self[r, c]
                if cell == SpecialCellValue.BLACK:
                    strings.append(cell.str * 3)
                else:
                    if numbers:
                        strings.append(str(cell.number) if cell.number else "")
                    else:
                        prefix = superscripts[cell.number % 10] if cell.number else " "
                        suffix = "*" if cell.shaded or cell.circled else " "
                        strings.append(f"{prefix}{cell.str}{suffix}")
            table.add_row(*strings)

        return table

    def to_text_grid(self, numbers: bool = False) -> str:
        """Returns a formatted text table representation of the crossword grid from Rust."""
        return self._inner.to_text_grid(numbers)

    def pprint(self, numbers: bool = False) -> None:
        """Prints a formatted string representation of the crossword fill."""
        console = Console()
        console.print(self._text_grid(numbers))

    def _repr_mimebundle_(
        self, include: Iterable[str], exclude: Iterable[str], **kwargs: Any
    ) -> dict[str, str]:
        html = self._grid_html()
        text = self._text_grid()._repr_mimebundle_([], [])["text/plain"]
        data = {"text/plain": text, "text/html": html}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data

    def _grid_html(self, size_px: int | None = None) -> str:
        """Returns an HTML rendering of the puzzle."""
        size_px = size_px or self.display_size_px
        suffix = token_hex(4)
        cell_size = size_px / max(self.num_rows, self.num_cols)
        cells = []
        for c in self.itercells():
            cell_number_span = f'<span class="cell-number">{c.number or ""}</span>'
            if c != BLACK:
                if len(c.str) > 1:
                    r_font_size = max(
                        int((cell_size * 0.85) / (len(c.str) * 0.55 + 0.4)),
                        6,
                    )
                    letter_span = f'<span class="letter rebus" style="font-size:{r_font_size}px;letter-spacing:-0.5px;">{c.str}</span>'
                else:
                    letter_span = f'<span class="letter">{c.str}</span>'
            else:
                letter_span = '<span class="letter"></span>'
            circle_span = '<span class="circle"></span>'
            if c == BLACK:
                extra_class = " black"
            elif c.shaded:
                extra_class = " gray"
            else:
                extra_class = ""
            cell_div = f"""
            <div class="crossword-cell{suffix}{extra_class}">
                {cell_number_span}
                {letter_span}
                {circle_span if c.circled else ""}
            </div>
            """
            cells.append(cell_div)
        val_font_size = max(int(cell_size * 0.55), 10)
        rebus_bottom = max(int(val_font_size * 0.38), 3)
        aspect_ratio = self.num_rows / self.num_cols
        css = CSS_TEMPLATE.format(
            num_cols=self.num_cols,
            height=size_px * min(1, aspect_ratio),
            width=size_px * min(1, 1 / aspect_ratio),
            num_font_size=max(int(cell_size * 0.28), 7),
            val_font_size=val_font_size,
            rebus_bottom=rebus_bottom,
            circle_dim=cell_size - 1,
            suffix=suffix,
        )
        cells_html = "\n".join(cells)
        return f"""
        <div>
            <style scoped>
                {css}
            </style>
            <div class="crossword{suffix}">
                {cells_html}
            </div>
        </div>
        """

    def check(
        self,
        symmetry: Symmetry | None = None,
        *,
        min_word_length: int = 3,
        allow_duplicates: bool = False,
        require_connected: bool = True,
        require_filled: bool = False,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validates the crossword puzzle against standard crossword rules.

        Checks:
        a) All word segments are at least `min_word_length` letters (no 1- or 2-letter fragments).
        b) Symmetry is satisfied (using `self.symmetry` or the provided `symmetry`).
        c) No words are reused across the puzzle (unless `allow_duplicates=True`).
        d) Full grid connectivity (all open squares form a single connected component).
        e) No empty cells if `require_filled=True`.
        """
        rust_sym = self._to_rust_sym(symmetry)
        is_valid, errors, warnings = self._inner.check(
            rust_sym,
            min_word_length,
            allow_duplicates,
            require_connected,
            require_filled,
        )

        result = ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
        if raise_on_error and not result.is_valid:
            raise ValueError(str(result))
        return result

    def is_valid(
        self,
        symmetry: Symmetry | None = None,
        *,
        min_word_length: int = 3,
        allow_duplicates: bool = False,
        require_connected: bool = True,
        require_filled: bool = False,
    ) -> bool:
        """Returns True if the crossword passes all validation rules, False otherwise."""
        return self.check(
            symmetry=symmetry,
            min_word_length=min_word_length,
            allow_duplicates=allow_duplicates,
            require_connected=require_connected,
            require_filled=require_filled,
        ).is_valid

    def stats(self) -> CrosswordStats:
        """Computes and returns crossword grid statistics."""
        data = self._inner.stats()
        return CrosswordStats(
            total_words=data["total_words"],
            across_words=data["across_words"],
            down_words=data["down_words"],
            black_squares=data["black_squares"],
            total_cells=data["total_cells"],
            open_cells=data["open_cells"],
            word_length_counts=data["word_length_counts"],
            letter_counts=data["letter_counts"],
            rebus_count=data["rebus_count"],
            circled_count=data["circled_count"],
            shaded_count=data["shaded_count"],
            filled_words=data["filled_words"],
            open_words=data["open_words"],
        )


def _parse_word_string_to_cell_values(
    cells: list[Cell], direction: Direction, value: str
) -> list[CellValue]:
    """Parses an input string to assign to a list of word slot cells.

    Supports:
      1. Explicit parenthesized rebus notation: "AB(FOO)CD"
      2. Plain strings where existing rebus cells consume multi-letter substrings: "ABFOOCD"
      3. Plain strings matching the cell count: "ABCDE"
    """
    num_cells = len(cells)
    clean_val = value.upper()

    # Case 1: Explicit parentheses syntax: "AB(FOO)CD"
    if "(" in clean_val and ")" in clean_val:
        tokens: list[str] = []
        i = 0
        while i < len(clean_val):
            if clean_val[i] == "(":
                j = clean_val.find(")", i)
                if j == -1:
                    raise ValueError(f"Unmatched parenthesis in word value: {value}")
                tokens.append(clean_val[i + 1 : j])
                i = j + 1
            else:
                tokens.append(clean_val[i])
                i += 1

        if len(tokens) == num_cells:
            result_parens: list[CellValue] = []
            for cell, tok in zip(cells, tokens):
                if len(tok) == 1:
                    if tok in SpecialCellValue.EMPTY.input_str_reprs:
                        result_parens.append(SpecialCellValue.EMPTY)
                    elif tok in SpecialCellValue.BLACK.input_str_reprs:
                        result_parens.append(SpecialCellValue.BLACK)
                    else:
                        result_parens.append(tok)
                else:
                    if isinstance(cell.value, Rebus):
                        if direction == Direction.ACROSS:
                            result_parens.append(
                                Rebus(across=tok, down=cell.value.down)
                            )
                        else:
                            result_parens.append(
                                Rebus(across=cell.value.across, down=tok)
                            )
                    else:
                        result_parens.append(Rebus(tok))
            return result_parens

    # Case 2: Matching current cell directional lengths (e.g. "ABFOOCD" when cell 2 has across len 3)
    cell_lens: list[int] = []
    rebus_indices = [i for i, c in enumerate(cells) if isinstance(c.value, Rebus)]
    for c in cells:
        if isinstance(c.value, Rebus):
            cell_lens.append(len(c.value.get_value(direction)))
        else:
            cell_lens.append(1)

    if rebus_indices and sum(cell_lens) == len(clean_val):
        result_rebus: list[CellValue] = []
        cursor = 0
        for cell, clen in zip(cells, cell_lens):
            chunk = clean_val[cursor : cursor + clen]
            cursor += clen
            if isinstance(cell.value, Rebus):
                if direction == Direction.ACROSS:
                    result_rebus.append(Rebus(across=chunk, down=cell.value.down))
                else:
                    result_rebus.append(Rebus(across=cell.value.across, down=chunk))
            else:
                if chunk in SpecialCellValue.EMPTY.input_str_reprs:
                    result_rebus.append(SpecialCellValue.EMPTY)
                elif chunk in SpecialCellValue.BLACK.input_str_reprs:
                    result_rebus.append(SpecialCellValue.BLACK)
                else:
                    result_rebus.append(chunk)
        return result_rebus

    # Case 3: Exactly 1 rebus cell in the word slot and len(clean_val) >= num_cells
    if len(rebus_indices) == 1 and len(clean_val) >= num_cells:
        k = rebus_indices[0]
        prefix_len = k
        suffix_len = num_cells - 1 - k
        mid_len = len(clean_val) - prefix_len - suffix_len

        result_single_rebus: list[CellValue] = []
        for idx in range(prefix_len):
            ch = clean_val[idx]
            if ch in SpecialCellValue.EMPTY.input_str_reprs:
                result_single_rebus.append(SpecialCellValue.EMPTY)
            elif ch in SpecialCellValue.BLACK.input_str_reprs:
                result_single_rebus.append(SpecialCellValue.BLACK)
            else:
                result_single_rebus.append(ch)

        mid_chunk = clean_val[prefix_len : prefix_len + mid_len]
        rebus_cell = cells[k]
        assert isinstance(rebus_cell.value, Rebus)
        if direction == Direction.ACROSS:
            result_single_rebus.append(
                Rebus(across=mid_chunk, down=rebus_cell.value.down)
            )
        else:
            result_single_rebus.append(
                Rebus(across=rebus_cell.value.across, down=mid_chunk)
            )

        suffix_start = prefix_len + mid_len
        for idx in range(suffix_len):
            ch = clean_val[suffix_start + idx]
            if ch in SpecialCellValue.EMPTY.input_str_reprs:
                result_single_rebus.append(SpecialCellValue.EMPTY)
            elif ch in SpecialCellValue.BLACK.input_str_reprs:
                result_single_rebus.append(SpecialCellValue.BLACK)
            else:
                result_single_rebus.append(ch)

        return result_single_rebus

    # Case 4: Exact character-to-cell length match
    if len(clean_val) == num_cells:
        result_simple: list[CellValue] = []
        for cell, ch in zip(cells, clean_val):
            if ch in SpecialCellValue.EMPTY.input_str_reprs:
                result_simple.append(SpecialCellValue.EMPTY)
            elif ch in SpecialCellValue.BLACK.input_str_reprs:
                result_simple.append(SpecialCellValue.BLACK)
            else:
                result_simple.append(ch)
        return result_simple

    raise ValueError(
        f"Value '{value}' (length {len(clean_val)}) does not match word slot length {num_cells}"
    )
