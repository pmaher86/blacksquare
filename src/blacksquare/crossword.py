from __future__ import annotations

import importlib
import io
from secrets import token_hex
from typing import Any, Iterable, Iterator, overload

import numpy as np
import puz
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
from blacksquare.symmetry import Symmetry
from blacksquare.types import (
    CellIndex,
    CellValue,
    Direction,
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

        if isinstance(value, SpecialCellValue):
            val_str = value.str
        elif isinstance(value, Cell):
            val_str = value.str
        else:
            val_str = str(value)
            if (
                len(val_str) != 1
                and val_str
                not in SpecialCellValue.BLACK.input_str_reprs
                + SpecialCellValue.EMPTY.input_str_reprs
            ):
                raise ValueError(f"Invalid cell value length: {val_str}")

        self._inner.set_cell_value(r, c, val_str)

    def set_word(self, word_index: WordIndex, value: str) -> None:
        """Sets a word to a new value.

        Args:
            word_index: The index of the word.
            value: The new value of the word.
        """
        if not isinstance(value, str):
            raise ValueError(f"Word value must be str, got {type(value)}")
        r_dir = self._to_rust_dir(word_index[0])
        num = int(word_index[1])
        try:
            self._inner.set_word_value(r_dir, num, str(value).upper())
        except ValueError as e:
            if "not found in grid" in str(e):
                raise IndexError(str(e))
            raise e

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
    ) -> Crossword | None:
        """Searches for a possible fill, and returns the result as a new Crossword
        object. Backed by the native Rust backtracking solver.

        Args:
            word_list: An optional word list to use instead of the default.
            timeout: The maximum time in seconds to search before returning.
            temperature: A parameter to control randomness.
            score_filter: A threshold to apply to the word list before filling.
            allow_repeats: Whether to allow duplicate words in the grid.

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
        )
        if filled_inner is not None:
            return Crossword(
                _inner=filled_inner,
                word_list=wl,
                display_size_px=self.display_size_px,
            )
        return None

    @classmethod
    def from_puz(cls, filename: str) -> Crossword:
        """Creates a Crossword object from a .puz file."""
        puz_obj = puz.read(filename)
        grid = np.reshape(
            list(puz_obj.solution),
            (puz_obj.height, puz_obj.width),
        )
        xw = cls(grid=grid)
        for cn in puz_obj.clue_numbering().across:
            xw[ACROSS, cn["num"]].clue = cn["clue"]
        for cn in puz_obj.clue_numbering().down:
            xw[DOWN, cn["num"]].clue = cn["clue"]
        return xw

    def to_puz(self, filename: str) -> None:
        """Outputs a .puz file from the Crossword object."""
        puz_black, puz_empty = ".", "-"
        puz_obj = puz.Puzzle()
        puz_obj.height = self.num_rows
        puz_obj.width = self.num_cols

        char_array = np.array([cell.str for cell in self.itercells()])
        puz_obj.solution = (
            "".join(char_array)
            .replace(EMPTY.str, puz_empty)
            .replace(BLACK.str, puz_black)
        )
        fill_grid = char_array.copy()
        fill_grid[fill_grid != BLACK.str] = puz_empty
        fill_grid[fill_grid == BLACK.str] = puz_black
        puz_obj.fill = "".join(fill_grid)
        sorted_words = sorted(
            list(self.iterwords()), key=lambda w: (w.number, w.direction)
        )
        puz_obj.clues = [w.clue for w in sorted_words]
        setattr(puz_obj, "cksum_global", puz_obj.global_cksum())
        setattr(puz_obj, "cksum_hdr", puz_obj.header_cksum())
        setattr(puz_obj, "cksum_magic", puz_obj.magic_cksum())
        puz_obj.save(filename)

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
                        strings.append(
                            f"{'^' if cell.number else ' '}{cell.str}{'*' if cell.shaded or cell.circled else ' '}"
                        )
            table.add_row(*strings)

        return table

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
        cells = []
        for c in self.itercells():
            cell_number_span = f'<span class="cell-number">{c.number or ""}</span>'
            letter_span = f'<span class="letter">{c.str if c != BLACK else ""}</span>'
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
        aspect_ratio = self.num_rows / self.num_cols
        cell_size = size_px / max(self.num_rows, self.num_cols)
        css = CSS_TEMPLATE.format(
            num_cols=self.num_cols,
            height=size_px * min(1, aspect_ratio),
            width=size_px * min(1, 1 / aspect_ratio),
            num_font_size=int(cell_size * 0.3),
            val_font_size=int(cell_size * 0.6),
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
