from __future__ import annotations

import copy
import io
from secrets import token_hex
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pdfkit
import puz
import PyPDF2
from tqdm.auto import tqdm

from blacksquare.cell import Cell
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
        word_list: WordList = DEFAULT_WORDLIST,
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
            word_list (WordList): The word list to use by default when finding
                solutions.
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

        self._numbers = np.zeros_like(self._grid, dtype=int)
        self._across = np.zeros_like(self._grid, dtype=int)
        self._down = np.zeros_like(self._grid, dtype=int)
        self._words = {}
        self._parse_grid()

        self.word_list = word_list
        self.display_size_px = display_size_px

    def __getitem__(self, key) -> str:
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], Direction) and is_intlike(key[1]):
                if key in [w.index for w in self.iterwords()]:
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
                needs_parse = self._grid[key] == BLACK or value == BLACK
                self._grid[key].value = value
                if needs_parse:
                    self._parse_grid()
            else:
                raise IndexError
        else:
            raise IndexError

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._grid = copy.deepcopy(self._grid)
        for cell in copied._grid.ravel():
            cell._parent = copied
        # Update word references
        copied._words = copy.deepcopy(self._words)
        for word in copied._words.values():
            word._parent = copied
        return copied

    def __repr__(self):
        return self._text_grid()

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
        puz_obj = puz.Puzzle()
        puz_obj.height = self.num_rows
        puz_obj.width = self.num_cols

        char_array = np.array([cell.str for cell in self._grid.ravel()])
        puz_obj.solution = "".join(char_array).replace(EMPTY.str, "-")
        fill_grid = char_array.copy()
        fill_grid[fill_grid != BLACK.str] = "-"
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

    def get_symmetric_index(self, index: CellIndex) -> CellIndex:
        """Gets the index of a symmetric grid cell. Useful for enforcing symmetry.

        Args:
            index (CellIndex): The input index.

        Returns:
            CellIndex: The index of the cell symmetric to the input.
        """
        return (self._num_rows - 1 - index[0], self._num_cols - 1 - index[1])

    def get_symmetric_word(self, word_index: WordIndex) -> Word:
        raise NotImplementedError

    def _parse_grid(self) -> None:
        """Updates all indices to reflect the state of the _grid property."""
        old_across, old_down = self._across, self._down
        padded = np.pad(self._grid, 1, constant_values=Cell(None, (None, None), BLACK))
        shifted_down, shifted_right = padded[:-2, 1:-1], padded[1:-1, :-2]

        is_open = ~np.equal(self._grid, BLACK)
        needs_num = (is_open) & (
            np.equal(shifted_down, BLACK) | np.equal(shifted_right, BLACK)
        )
        self._numbers = np.reshape(np.cumsum(needs_num), self._grid.shape) * needs_num
        self._across = np.maximum.accumulate(
            np.equal(shifted_right, BLACK) * self._numbers, axis=1
        ) * (is_open)
        self._down = np.maximum.accumulate(
            np.equal(shifted_down, BLACK) * self._numbers
        ) * (is_open)

        affected_across = (
            set(self._across[old_across != self._across].flatten())
            | set(old_across[old_across != self._across].flatten())
        ) - {0}
        affected_down = (
            set(self._down[old_down != self._down].flatten())
            | set(old_down[old_down != self._down].flatten())
        ) - {0}
        # Remove words that don't exist anymore
        removed = set()
        curr_word_keys = list(self._words.keys())
        for word_key in curr_word_keys:
            if word_key[1] not in self._get_direction_mask(word_key[0]):
                del self._words[word_key]
                removed.add(word_key)

        # Update words
        for across_num in affected_across - {n for d, n in removed if d == ACROSS}:
            self._words[(ACROSS, across_num)] = Word(self, ACROSS, across_num)
        for down_num in affected_down - {n for d, n in removed if d == DOWN}:
            self._words[(DOWN, down_num)] = Word(self, DOWN, down_num)

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

    def iterwords(self, direction: Optional[Direction] = None) -> Iterator[Word]:
        """Method for iterating over the words in the crossword.

        Args:
            direction (Direction, optional): If provided, limits the iterator to only
                the given direction.

        Yields:
            Iterator[Word]: An iterator of Word objects. Ordered in standard crossword
            fashion (ascending numbers, across then down).
        """
        for word_index in sorted(self._words.keys()):
            if direction is None or direction == self[word_index].direction:
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

    def find_solutions(
        self,
        word_indices: List[WordIndex],
        word_list: Optional[WordList] = None,
        beam_width=100,
    ) -> List[Crossword]:
        """Finds solutions for the provided words. Solutions are returned as new
        crossword objects.

        Args:
            word_indices (List[WordIndex]): The words to consider.
            word_list (Optional[WordList], optional): The word list to use. If None, the
                default word list for the crossword will be use.
            beam_width (int, optional): Search parameter, how many branches to consider.
                Defaults to 100.

        Returns:
            List[Crossword]: A list of solutions, ranked by score.
        """
        word_list = self.word_list if word_list is None else word_list
        memory = [self]
        sorted_indices = sorted(
            word_indices, key=lambda i: len(word_list.find_matches(self[i]))
        )
        for word_index in tqdm(sorted_indices):
            new_memory = []
            for xw in memory:
                matches = xw[word_index].find_matches(word_list=word_list)
                for word, score in zip(
                    matches.words[:beam_width], matches.scores[:beam_width]
                ):
                    new_memory.append((xw, word, score))
            memory = [
                xw.set_word(word_index, w, inplace=False)
                for xw, w, s in sorted(new_memory, key=lambda x: x[2], reverse=True)[
                    :beam_width
                ]
            ]
        scored_memory = [
            (
                xw,
                np.product(
                    [word_list.get_score(xw[idx].value) for idx in word_indices]
                ),
            )
            for xw in memory
        ]
        return [
            xw for xw, score in sorted(scored_memory, key=lambda x: x[1], reverse=True)
        ]

    def find_area_solutions(
        self, seed_word_index: WordIndex, word_list: Optional[WordList] = None
    ) -> List[Crossword]:
        """A method to solve entire contiguous areas of the crossword, starting from a
        seed.

        Args:
            seed_word_index (WordIndex): The word to start from.
            word_list (Optional[WordList], optional): The word list to use. If None, the
                default word list for the crossword will be use.

        Returns:
            List[Crossword]: A list of solutions, ranked by score.
        """
        word_list = self.word_list if word_list is None else word_list
        open_words = self.get_contiguous_open_words(seed_word_index)
        return self.find_solutions([w.index for w in open_words], word_list=word_list)

    def get_contiguous_open_words(self, seed_word_index: WordIndex) -> List[Word]:
        """Gets the words representing a contiguous open area of the crossword grid.

        Args:
            seed_word_index (WordIndex): The word to start from.

        Returns:
            List[Word]: A list of all open words connected to the seed word.
        """
        open_words = {self[seed_word_index]}
        while True:
            adjacent_open_words = set()
            for word in open_words:
                adjacent_open_words.update(
                    cross
                    for cell, cross in zip(word.cells, word.crosses)
                    if cross.is_open() and cell == EMPTY
                )
            new_open_words = open_words | adjacent_open_words
            if new_open_words == open_words:
                break
            else:
                open_words = new_open_words
        return sorted(open_words, key=lambda w: w.index)

    # TODO: Implement depth-first search.

    def _text_grid(self, numbers: bool = False) -> str:
        """Returns a formatted string representation of the crossword fill.

        Args:
            numbers (bool): If True, prints the numbers in the grid rather
                than the letters. Defaults to False.

        Returns:
            str: A formatted string representation of the crossword.
        """
        out_str = ["┌" + "───┬" * (self._num_cols - 1) + "───┐"]
        for i in range(self._num_rows):
            row_string = "│"
            for j in range(self._num_cols):
                if self._grid[i, j] == BLACK:
                    value = "███"
                elif not numbers:
                    value = f" {self._grid[i,j].str} "
                elif self._numbers[i, j]:
                    value = f"{self._numbers[i,j]: <3d}"
                else:
                    value = "   "
                row_string += value + "│"
            out_str.append(row_string)
            if i < self._num_rows - 1:
                out_str.append("├" + "───┼" * (self._num_cols - 1) + "───┤")
        out_str.append("└" + "───┴" * (self._num_cols - 1) + "───┘")
        return "\n".join(out_str)

    def pprint(self, numbers: bool = False) -> str:
        """Prints a formatted string representation of the crossword fill.

        Args:
            numbers (bool): If True, prints the numbers in the grid rather
                than the letters. Defaults to False.
        """
        print(self._text_grid(numbers))

    def _repr_html_(self) -> str:
        return self._grid_html()

    def _grid_html(self, size_px: Optional[int] = None) -> str:
        """Returns an HTML rendering of the puzzle.

        Args:
            size_px (int): The size of the largest dimension in pixels. If None
                provided, defaults to the display_size_px property.

        Returns:
            str: HTML to display the puzzle.
        """
        # Random suffix is a hack to ensure correct display in Jupyter settings
        size_px = size_px or self.display_size_px
        suffix = token_hex(4)
        row_elems = []
        for r in range(self._num_rows):
            cells = []
            for c in range(self._num_cols):
                number = self._numbers[r, c]
                cells.append(
                    f"""<td class='xw{suffix}{ f" black{suffix}" if self._grid[r, c] == BLACK else ""}'>
                        <div class='number{suffix}'> {number if number else ""}</div>
                        <div class='value{suffix}'>{self._grid[r,c].str if self._grid[r,c] != BLACK else ""}</div>
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
