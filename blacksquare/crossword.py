from __future__ import annotations

import copy
import io
import time
from secrets import token_hex
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pdfkit
import puz
import PyPDF2
import rich.box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from blacksquare.cell import Cell
from blacksquare.symmetry import Symmetry
from blacksquare.types import CellIndex, Direction, SpecialCellValue, WordIndex
from blacksquare.utils import is_intlike
from blacksquare.word import Word
from blacksquare.word_list import DEFAULT_WORDLIST, WordList

BLACK, EMPTY = SpecialCellValue.BLACK, SpecialCellValue.EMPTY
ACROSS, DOWN = Direction.ACROSS, Direction.DOWN


class Crossword:
    """An object representing a crossword puzzle."""

    def __init__(
        self,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        grid: Optional[Union[List[List[str]], np.ndarray]] = None,
        symmetry: Optional[Symmetry] = Symmetry.ROTATIONAL,
        word_list: Optional[WordList] = None,
        display_size_px: int = 450,
    ):
        """Creates a new Crossword object.

        Args:
            num_rows (int, optional): The number of rows in the puzzle. Either this or
                a grid must be provided. If grid is provided, shape will be inferred.
            num_cols (int, optional): The number of columns in the puzzle. If None, it
                will either be equal to number of rows or inferred from grid.
            grid (Union[List[List[str]], np.ndarray]], optional): A 2-D array of
                letters from which the grid will be initialized. Can be provided
                instead of num_rows/num_cols.
            word_list (WordList, optional): The word list to use by default when finding
                solutions. If None, defaults to the default word list.
            display_size_px (int): The size in pixels of the largest dimension of the
                puzzle HTML rendering.
        """
        assert (num_rows is not None) ^ (
            grid is not None
        ), "Either specify shape or provide grid."
        if num_rows:
            self._num_rows = num_rows
            if num_cols:
                self._num_cols = num_cols
            else:
                self._num_cols = self._num_rows

            shape = (self._num_rows, self._num_cols)
            cells = [Cell(self, (i, j)) for i, j in np.ndindex(*shape)]
            self._grid = np.array(cells, dtype=object).reshape(shape)
        elif grid is not None:
            assert np.all([len(r) == len(grid[0]) for r in grid])
            self._num_rows = len(grid)
            self._num_cols = len(grid[0])
            shape = (self._num_rows, self._num_cols)
            cells = [Cell(self, (i, j), grid[i][j]) for i, j in np.ndindex(*shape)]
            self._grid = np.array(cells, dtype=object).reshape(shape)

        if symmetry.requires_square and self._num_rows != self._num_cols:
            raise ValueError(f"{symmetry.value} symmetry requires a square grid.")

        self._numbers = np.zeros_like(self._grid, dtype=int)
        self._across = np.zeros_like(self._grid, dtype=int)
        self._down = np.zeros_like(self._grid, dtype=int)
        self._words = {}
        self._parse_grid()

        self.word_list = word_list if word_list is not None else DEFAULT_WORDLIST
        self.display_size_px = display_size_px
        self.symmetry = symmetry

    def __getitem__(self, key) -> str:
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], Direction) and is_intlike(key[1]):
                if key in self._words:
                    return self._words[key]
                else:
                    raise IndexError
            elif is_intlike(key[0]) and is_intlike(key[1]):
                return self._grid[key]
        raise IndexError

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], Direction) and is_intlike(key[1]):
                if not isinstance(value, str) or len(self[key]) != len(value):
                    raise ValueError
                for letter, cell in zip(value, self._grid[self._get_word_mask(key)]):
                    cell.value = letter
            elif is_intlike(key[0]) and is_intlike(key[1]):
                if value == BLACK:
                    self._grid[key].value = BLACK
                    images = self.get_symmetric_cell_index(key, force_list=True)
                    for image in images:
                        self._grid[image].value = BLACK
                    self._parse_grid()
                elif self._grid[key] == BLACK:
                    self._grid[key].value = value
                    images = self.get_symmetric_cell_index(key, force_list=True)
                    for image in images:
                        if self._grid[image].value == BLACK:
                            self._grid[image].value = EMPTY
                    self._parse_grid()
                else:
                    self._grid[key].value = value
            else:
                raise IndexError
        else:
            raise IndexError

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._grid = copy.deepcopy(self._grid)
        for cell in copied._grid.ravel():
            cell._parent = copied
        copied._words = copy.deepcopy(self._words)
        for word in copied._words.values():
            word._parent = copied
        return copied

    def __repr__(self):
        longest_filled_word = max(
            self.iterwords(), key=lambda w: len(w) if not w.is_open() else 0
        )
        return f'Crossword("{longest_filled_word.value}")'

    @classmethod
    def from_puz(cls, filename: str) -> Crossword:
        """Creates a Crossword object from a .puz file.

        Args:
            filename (str): The path of the input .puz file.

        Returns:
            Crossword: A Crossword object.
        """
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
        """Outputs a .puz file from the Crossword object.

        Args:
            filename (str): The output path.
        """
        puz_black, puz_empty = ".", "-"
        puz_obj = puz.Puzzle()
        puz_obj.height = self.num_rows
        puz_obj.width = self.num_cols

        char_array = np.array([cell.str for cell in self._grid.ravel()])
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
        puz_obj.cksum_global = puz_obj.global_cksum()
        puz_obj.cksum_hdr = puz_obj.header_cksum()
        puz_obj.cksum_magic = puz_obj.magic_cksum()
        puz_obj.save(filename)

    def to_pdf(
        self,
        filename: str,
        header: Optional[List[str]] = None,
    ) -> None:
        """Outputs a .pdf file in NYT submission format from the Crossword object.

        Args:
            filename (str): The output path.
            header (List[str], optional): A list of strings to put on the output (e.g.
                name, address, etc.). Each list element will be one line in the header.
        """

        header_html = "<br />".join(header) if header else ""
        grid_html = f"""
            <html>
            <head><meta charset="utf-8"></head>
            <body>
            <div style='font-size:18pt;'>
                {header_html}
            </div>
            <div style='position:absolute;left:50%;top:50%;transform: translate(-50%, -50%);'> 
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
                        font-size:18pt;
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
        merger = PyPDF2.PdfFileMerger()
        for html_page in [grid_html, clue_html]:
            pdf = pdfkit.from_string(
                html_page,
                False,
                options={
                    "quiet": None,
                    "margin-top": "0.5in",
                    "margin-right": "0.5in",
                    "margin-bottom": "0.5in",
                    "margin-left": "0.5in",
                    "encoding": "UTF-8",
                },
            )
            merger.append(PyPDF2.PdfFileReader(io.BytesIO(pdf)))
        merger.write(str(filename))
        merger.close()

    @property
    def num_rows(self) -> int:
        """int: The number of rows in the puzzle"""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """int: The number of columns in the puzzle"""
        return self._num_cols

    @property
    def clues(self) -> Dict[WordIndex, str]:
        """Dict[WordIndex, str]: A dict mapping word index to clue."""
        return {index: w.clue for index, w in self._words.items()}

    def get_symmetric_cell_index(
        self, index: CellIndex, force_list: bool = False
    ) -> Optional[Union[CellIndex, List[CellIndex]]]:
        """Gets the index of a symmetric grid cell. Useful for enforcing symmetry.

        Args:
            index (CellIndex): The input index.

        Returns:
            CellIndex: The index of the cell symmetric to the input.
        """
        if not self.symmetry:
            return [] if force_list else None
        elif self.symmetry.is_multi_image:
            results = self.symmetry.apply(self._grid)
            return list({r.grid[index].index for r in results})
        else:
            image = self.symmetry.apply(self._grid).grid[index].index
            return [image] if force_list else image

    def get_symmetric_word_index(
        self, word_index: WordIndex, force_list: bool = False
    ) -> Optional[Union[WordIndex, List[WordIndex]]]:
        dir = word_index[0]
        mask = self._get_word_mask(word_index)
        if not self.symmetry:
            return [] if force_list else None
        elif self.symmetry.is_multi_image:
            results = self.symmetry.apply(self._grid)
            new_indices = set()
            for result in results:
                new_dir = dir.opposite if result.word_direction_rotated else dir
                new_indices.add(result.grid[mask][0].get_parent_word(new_dir).index)
            return list(new_indices)
        else:
            result = self.symmetry.apply(self._grid)
            new_dir = dir.opposite if result.word_direction_rotated else dir
            image = result.grid[mask][0].get_parent_word(new_dir).index
            return [image] if force_list else image

    def _parse_grid(self) -> None:
        """Updates all indices to reflect the state of the _grid property."""
        old_across, old_down = self._across, self._down
        padded = np.pad(self._grid, 1, constant_values=Cell(None, (None, None), BLACK))
        shifted_down, shifted_right = padded[:-2, 1:-1], padded[1:-1, :-2]

        is_open = ~np.equal(self._grid, BLACK)
        starts_down, starts_across = (
            np.equal(x, BLACK) for x in (shifted_down, shifted_right)
        )
        needs_num = (is_open) & (starts_down | starts_across)
        self._numbers = np.reshape(np.cumsum(needs_num), self._grid.shape) * needs_num
        self._across = (
            np.maximum.accumulate(starts_across * self._numbers, axis=1) * is_open
        )
        self._down = np.maximum.accumulate(starts_down * self._numbers) * is_open

        def get_cells_to_nums(ordered_nums: np.ndarray) -> Dict[Tuple[int, ...], int]:
            flattened = ordered_nums.ravel()
            word_divs = np.flatnonzero(np.diff(flattened, prepend=-1))
            nums = flattened[word_divs]
            groups = np.split(np.arange(len(flattened)), word_divs[1:])
            return dict(zip(map(tuple, groups), nums))

        def get_new_to_old_map(old: np.ndarray, new: np.ndarray) -> Dict[int, int]:
            old_cells_nums = get_cells_to_nums(old)
            new_cells_nums = get_cells_to_nums(new)
            new_to_old = {}
            for cells in set(old_cells_nums.keys()).intersection(new_cells_nums.keys()):
                if old_cells_nums[cells] and new_cells_nums[cells]:
                    new_to_old[new_cells_nums[cells]] = old_cells_nums[cells]
            return new_to_old

        across_new_old_map = get_new_to_old_map(old_across, self._across)
        down_new_old_map = get_new_to_old_map(old_down.T, self._down.T)

        new_words = {}
        for across_num in set(self._across.ravel()) - {0}:
            old_word = self._words.get((ACROSS, across_new_old_map.get(across_num)))
            new_words[(ACROSS, across_num)] = Word(
                self,
                ACROSS,
                across_num,
                clue=old_word.clue if old_word is not None else "",
            )
        for down_num in set(self._down.ravel()) - {0}:
            old_word = self._words.get((DOWN, down_new_old_map.get(down_num)))
            new_words[(DOWN, down_num)] = Word(
                self, DOWN, down_num, clue=old_word.clue if old_word is not None else ""
            )
        self._words = new_words

    def _get_direction_mask(self, direction: Direction) -> np.ndarray:
        """A boolean mask indicating the word number for each cell for a given
        direction.

        Args:
            direction (Direction): The desired direction.

        Returns:
            np.ndarray: The grid of word numbers for each cell.
        """
        if direction == Direction.ACROSS:
            return self._across
        elif direction == Direction.DOWN:
            return self._down

    def _get_word_mask(self, word_index: WordIndex) -> np.ndarray:
        """A boolean mask that indicates which grid cells belong to a word.

        Args:
            word_index (WordIndex): The index of the desired word.

        Returns:
            np.ndarray: The grid indicating which cells are in the input word.
        """
        word = self[word_index]
        return self._get_direction_mask(word.direction) == word.number

    def get_word_cells(self, word_index: WordIndex) -> List[Cell]:
        return list(self._grid[self._get_word_mask(word_index)])

    def get_cell_number(self, cell_index: CellIndex) -> Optional[int]:
        """Gets the crossword numeral at a given cell, if it exists.

        Args:
            cell_index (CellIndex): The index of the cell.

        Returns:
            Optional[int]: The crossword number in that cell, if any.
        """
        number = self._numbers[cell_index]
        if number:
            return number

    def iterwords(
        self, direction: Optional[Direction] = None, only_open: bool = False
    ) -> Iterator[Word]:
        """Method for iterating over the words in the crossword.

        Args:
            direction (Direction, optional): If provided, limits the iterator to only
                the given direction.
            only_open(bool): Whether to only return open words. Defaults to False.

        Yields:
            Iterator[Word]: An iterator of Word objects. Ordered in standard crossword
            fashion (ascending numbers, across then down).
        """
        for word_index in sorted(self._words.keys()):
            if direction is None or direction == self[word_index].direction:
                if not only_open or self._words[word_index].is_open():
                    yield (self._words[word_index])

    def itercells(self) -> Iterator[Cell]:
        """Method for iterating over the cells in the crossword.

        Yields:
            Iterator[Cell]: An iterator of Cell objects. Ordered left to right, top to
            bottom.
        """
        for cell in self._grid.ravel():
            yield cell

    def get_indices(self, word_index: WordIndex) -> List[CellIndex]:
        """Gets the list of cell indices for a given word.

        Args:
            word_index (WordIndex): The index of the desired word.

        Returns:
            List[CellIndex]: A list of cell indices that belong to the word.
        """
        return [
            (int(x[0]), int(x[1]))
            for x in np.argwhere(self._get_word_mask(word_index)).tolist()
        ]

    def get_word_at_index(
        self, index: CellIndex, direction: Direction
    ) -> Optional[Word]:
        """Gets the word that passes through a cell in a given direction.

        Args:
            index (CellIndex): The index of the cell.
            direction (Direction): The direction of the word.

        Returns:
            Optional[Word]: The word passing through the index in the provided
            direction. If the index corresponds to a black square, this method returns
            None.
        """
        if self[index] != BLACK:
            number = self._get_direction_mask(direction)[index]
            return self[direction, number]

    def set_word(
        self, word_index: WordIndex, value: str, inplace: bool = True
    ) -> Optional[Crossword]:
        """Sets a word to a new value. If inplace is set to False, returns a new
        Crossword object rather than modifying the current one.

        Args:
            word_index (WordIndex): The index of the word.
            value (str): The new value of the word.
            inplace (bool): Whether to modify the current object.

        Returns:
            Optional[Crossword]: If inplace is False, a new Crossword object is
            returned with new value.
        """
        if inplace:
            self[word_index] = value
        else:
            new_crossword = copy.deepcopy(self)
            new_crossword[word_index] = value
            return new_crossword

    def set_cell(
        self, index: CellIndex, value: str, inplace: bool = True
    ) -> Optional[Crossword]:
        """Sets a cell to a new value. If inpace is set to False, returns a new
        Crossword object rather than modifying the current one.

        Args:
            index (CellIndex): The index of the cell.
            value (str): The new value of the cell.
            inplace (bool): Whether to modify the current object.

        Returns:
            Optional[Crossword]: If inplace is False, a new Crossword object is
            returned with new value.
        """
        if inplace:
            self[index] = value
        else:
            new_crossword = copy.deepcopy(self)
            new_crossword[index] = value
            return new_crossword

    def fill(
        self,
        word_list: Optional[WordList] = None,
        timeout: Optional[int] = 30,
        temperature: float = 0,
    ) -> Optional[Crossword]:
        """Searches for a possible fill, and returns the result as a new Crossword
        object. Uses a modified depth-first-search algorithm.

        Args:
            timeout (int, optional): The maximum time in seconds to search before
                returning. Defaults to 30. If None, will search until completion.
            temperature (float, optional): A parameter to control randomness. Defaults
                to 0 (no randomness). Reasonable values are around 1.

        Returns:
            Optional[Crossword]: The filled Crossword. Returns None if the search is
                exhausted or the timeout is hit.
        """
        start_time = time.time()
        word_list = word_list if word_list is not None else self.word_list

        def recurse_solve(xw: Crossword, display_context: Live) -> Optional[Crossword]:
            open_words = [w for w in xw.iterwords() if w.is_open()]
            if len(open_words) == 0:
                return xw
            num_matches = np.array([len(word_list.find_matches(w)) for w in open_words])
            noise = np.abs(np.random.normal(scale=num_matches)) * temperature
            word_to_match = open_words[np.argmin(num_matches + noise)]
            matches = word_to_match.find_matches(word_list)
            if len(matches) == 0:
                return
            else:
                noisy_matches = matches.rescore(
                    lambda _, s: s * np.random.lognormal(0.0, 0.1 * temperature)
                )
                new_xw = copy.deepcopy(xw)
                for match in noisy_matches.words:
                    if timeout and time.time() > start_time + timeout:
                        return
                    new_xw[word_to_match.index] = match
                    display_context.update(new_xw._text_grid())
                    fill = recurse_solve(new_xw, live)
                    if fill:
                        return fill

        with Live(self._text_grid(), refresh_per_second=4, transient=True) as live:
            solution = recurse_solve(self, live)
            if solution is not None:
                live.update(solution._text_grid(), refresh=True)

        return solution

    def _text_grid(self, numbers: bool = False) -> Table:
        """Returns a rich Table that displays the crossword.

        Args:
            numbers (bool): If True, prints the numbers in the grid rather
                than the letters. Defaults to False.

        Returns:
            Table: A Table object containing the crossword.
        """
        table = Table(
            box=rich.box.SQUARE,
            show_header=False,
            show_lines=True,
            width=4 * self.num_cols + 1,
            padding=0,
        )
        for c in range(self.num_cols):
            table.add_column(justify="left", width=3)
        for row in self._grid:
            strings = []
            for cell in row:
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

    def pprint(self, numbers: bool = False) -> str:
        """Prints a formatted string representation of the crossword fill.

        Args:
            numbers (bool): If True, prints the numbers in the grid rather
                than the letters. Defaults to False.
        """
        console = Console()
        console.print(self._text_grid(numbers))

    def _repr_mimebundle_(
        self, include: Iterable[str], exclude: Iterable[str], **kwargs: Any
    ) -> Dict[str, str]:
        """A display method that handles different IPython environments.

        Args:
            include (Iterable[str]): MIME types to include.
            exclude (Iterable[str]): MIME types to exclude.

        Returns:
            Dict[str, str]: A dict containing the outputs.
        """

        html = self._grid_html()
        text = self._text_grid()._repr_mimebundle_([], [])["text/plain"]
        data = {"text/plain": text, "text/html": html}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data

    def _grid_html(self, size_px: Optional[int] = None) -> str:
        """Returns an HTML rendering of the puzzle.

        Args:
            size_px (int): The size of the largest dimension in pixels. If None
                provided, defaults to the display_size_px property.

        Returns:
            str: HTML to display the puzzle.
        """
        size_px = size_px or self.display_size_px
        # Random suffix is a hack to ensure correct display in Jupyter settings
        suffix = token_hex(4)
        circle_string = f"<div class='circle{suffix}'> </div>"
        row_elems = []
        for r in range(self._num_rows):
            cells = []
            for c in range(self._num_cols):
                number = self._numbers[r, c]
                cells.append(
                    f"""<td class='xw{suffix}{f" black{suffix}" if self._grid[r, c] == BLACK else ""}{f" gray{suffix}" if self._grid[r, c].shaded else ""}'>
                        <div class='number{suffix}'> {number if number else ""}</div>
                        <div class='value{suffix}'>
                            {self._grid[r,c].str if self._grid[r,c] != BLACK else ""}
                        </div>
                        {circle_string if self._grid[r,c].circled else ""}
                    </td >"""
                )
            row_elems.append(f"<tr class='xw{suffix}'>{''.join(cells)}</tr>")
        aspect_ratio = self.num_rows / self.num_cols
        cell_size = size_px / max(self.num_rows, self.num_cols)
        return """
        <div>
        <style scoped>
        table.xw{suffix} {{table-layout:fixed; background-color:white;width:{width}px;height:{height}px;}}
        td.xw{suffix} {{outline: 2px solid black;outline-offset: -1px;position: relative;font-family: Arial, Helvetica, sans-serif;}}
        tr.xw{suffix} {{background-color: white !important;}}
        .number{suffix} {{position: absolute;top: 2px;left: 2px;font-size: {num_font}px;font-weight: normal;user-select: none;}}
        .value{suffix} {{position: absolute;bottom:0;left: 50%;font-weight: bold;font-size: {val_font}px; transform: translate(-50%, 0%);}}
        .black{suffix} {{background-color: black;}}
        .gray{suffix} {{background-color: lightgrey;}}
        .circle{suffix} {{position: absolute; border-radius: 50%; border: 1px solid black; right: 0px; left: 0px; top: 0px; bottom: 0px;}}
        </style>
        <table class='xw{suffix}'><tbody>
            {rows}
        </tbody</table>
        </div>
        """.format(
            height=size_px * min(1, aspect_ratio),
            width=size_px * min(1, 1 / aspect_ratio),
            rows="\n".join(row_elems),
            suffix=suffix,
            num_font=int(cell_size * 0.3),
            val_font=int(cell_size * 0.6),
        )
