import copy
import enum
import io
from typing import Dict, Iterator, List, Optional, Tuple, Union
from secrets import token_hex

import numpy as np
import pdfkit
import puz
import PyPDF2
from frozendict import frozendict
from tqdm.auto import tqdm

from .dictionary import (
    Dictionary,
    cached_regex_match,
    load_default_dictionary,
    parse_dictionary,
)
from .utils import sum_by_group

_DEFAULT_DICT = load_default_dictionary()


class Direction(enum.Enum):
    """An Enum representing the directions of words in a crossword."""

    ACROSS = "Across"
    DOWN = "Down"

    @property
    def opposite(self) -> "Direction":
        if self == Direction.ACROSS:
            return Direction.DOWN
        else:
            return Direction.ACROSS

    def __repr__(self):
        return f"<{self.value}>"


black = "#"
empty = " "
across = Direction.ACROSS
down = Direction.DOWN


class Word:
    """An object representing a single Word, with awares of the parent grid."""

    def __init__(
        self,
        parent_crossword: "Crossword",
        direction: Direction,
        number: int,
        clue: str = "",
    ):
        """Retuns a new Word object.

        Args:
            parent_crossword (Crossword): The parent crossword for the word.
            direction (Direction): The direction of the word.
            number (int): The ordinal number of the word.
            clue (str, optional): The clue associated with the word. Defaults to "".
        """
        self.clue = clue
        self._parent = parent_crossword
        self._direction = direction
        self._number = number

    def __getitem__(self, key) -> str:
        if not isinstance(key, int):
            raise IndexError
        return self.value[key]

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise IndexError
        grid_indices = self._parent.get_indices(self.index)
        self._parent[tuple(grid_indices[key])] = value

    @property
    def direction(self) -> Direction:
        """Direction: The direction of the word."""
        return self._direction

    @property
    def number(self) -> int:
        """int: The number of the word."""
        return self._number

    @property
    def index(self) -> Tuple[Direction, int]:
        """Tuple[Direction, int]: The (direction, number) index of the word."""
        return (self.direction, self.number)

    @property
    def length(self) -> int:
        """int: The number of crossword cells in the word."""
        return len(self.value)

    @property
    def value(self) -> str:
        """str: The current fill value of the word"""
        return "".join(self._parent.grid[self._parent._get_word_mask(self.index)])

    @value.setter
    def value(self, new_value: str):
        self._parent[self.index] = new_value.upper()

    def is_open(self) -> bool:
        """Does the word contain any blank spaces.

        Returns:
            bool: True if any of the letters are blank.
        """
        return np.any(np.array(list(self.value)) == empty)

    def get_crosses(self) -> List["Word"]:
        """Returns the words that cross the current word.

        Returns:
            list[Word]: A list of Word objects corresponding to the crosses.
        """
        indices = self._parent.get_indices(self.index)
        crosses = []
        for index in indices:
            words = self._parent.get_words_at_index(index)
            for word in words:
                if word.direction == self.direction.opposite:
                    crosses.append(word)
        return crosses

    def _find_matches_np(self, dictionary: frozendict) -> np.ndarray:
        """A helper method to find matches using caching.

        Args:
            dictionary (frozendict): The dictionary to use.

        Returns:
            np.ndarray: An array of matching words from the dictionary.
        """
        regex = "^" + "".join(["." if c == empty else c for c in self.value]) + "$"
        return cached_regex_match(dictionary, regex)

    def find_matches(
        self,
        sort_method="alpha",
        dictionary: Optional[Dictionary] = None,
        return_scores=False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Finds matches for the current word from a dictionary. Uses various methods
        to rank the outputs:
            * "alpha" sorts matches alphabetically
            * "score" sorts matches by the score they have in the dictionary
            * "cross_match" sorts matches by the product of the sume of available
              matches at each open letter times the word's score.

        "cross_match" is the slowest of the three methods but provides the most useful
        results.

        Args:
            sort_method (str, optional): The method of sorting. Can be one of ("alpha",
                "score", "cross_match"). Defaults to "alpha".
            dictionary (Optional[Dictionary], optional): A dictionary to use. Can be a
                dict with scores or a list of words (all scores will default to 1). If
                None provided, the Crossword's default dict will be used.
            return_scores (bool, optional): Whether to return the associated scores
                with the words. Only used for "score" and "cross_match" sort methods.
                Defaults to False.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: If return_scores is False,
                returns a list of matches sorted by the chosen criteria. If
                return_scores is True, retuns a sorted list of (word, score) tuples.
        """
        dictionary = (
            parse_dictionary(dictionary) if dictionary else self._parent._dictionary
        )
        matches = self._find_matches_np(dictionary)
        if sort_method == "alpha":
            scores = matches
        elif sort_method == "score":
            scores = [dictionary[m] for m in matches]
        elif sort_method == "cross_match":
            scores = self._cross_match_scores(matches, dictionary)
        else:
            raise ValueError("Invalid sort method")
        sorted_matches = sorted(
            [(w, s) for w, s in zip(matches, scores)],
            key=lambda x: x[1],
            reverse=sort_method != "alpha",
        )
        if return_scores and sort_method != "alpha":
            return sorted_matches
        else:
            return [s[0] for s in sorted_matches]

    def _cross_match_scores(
        self, matches: np.ndarray, dictionary: frozendict
    ) -> np.ndarray:
        """Scores the list of matches uing the cross_match method. The heuristic aims
        to maximize the number of good words allowed by the crosses.

        Args:
            matches (np.ndarray): A numpy array of matching words.
            dictionary (frozendict): The parsed dictionary to use for cross matches.

        Returns:
            np.ndarray: A numpy array of scores, corresponding to the input matches.
        """
        word_scores = np.vectorize(dictionary.get, otypes=[float])
        open_indices = np.argwhere(np.array(list(self.value)) == empty).squeeze(1)
        crosses = self.get_crosses()
        letter_scores_per_index = {}
        for index in open_indices:
            cross = crosses[index]
            cross_index = cross.get_crosses().index(self)
            cross_matches = cross._find_matches_np(dictionary)
            if len(cross_matches) > 0:
                cross_scores = word_scores(cross_matches)
                match_letters = cross_matches.view("U1").reshape(
                    len(cross_matches), -1
                )[:, cross_index]
                letter_scores_per_index[index] = sum_by_group(
                    match_letters, cross_scores
                )
            else:
                letter_scores_per_index[index] = {}

        score_word_fn = np.vectorize(
            lambda w: np.prod(
                [letter_scores_per_index[i].get(w[i], 0) for i in open_indices]
            )
            * dictionary[w],
            otypes=[float],
        )
        scores = score_word_fn(matches)
        return scores

    def __repr__(self):
        return f'{self.direction.value} {self.number}: "{self.value.replace(" ", "?")}"'


class Crossword:
    """An object reprsenting a crossword puzzle."""

    def __init__(
        self,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        grid: Optional[Union[List[List[str]], np.ndarray]] = None,
        dictionary: Optional[Dictionary] = None,
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
            dictionary (Union[Dict[str, float], List[str]], optional): A dictionary to
                use as the default for finding matches. If None, Peter Broda's scored
                wordlist will be used.
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
            self._grid = np.array(
                [empty for _ in range(self._num_rows * self._num_cols)]
            ).reshape((self._num_rows, self._num_cols))
        elif grid is not None:
            self._grid = np.array(grid, dtype="U1")
            self._num_rows, self._num_cols = self._grid.shape

        self._numbers = np.zeros_like(self._grid, dtype=int)
        self._across = np.zeros_like(self._grid, dtype=int)
        self._down = np.zeros_like(self._grid, dtype=int)
        self._words = {}
        self._parse_grid()
        if dictionary:
            self._dictionary = parse_dictionary(dictionary)
        else:
            self._dictionary = _DEFAULT_DICT
        self.display_size_px = display_size_px

    def __getitem__(self, key) -> str:
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], Direction) and isinstance(key[1], int):
                if key in [w.index for w in self.iterwords()]:
                    return self._words[key]
                else:
                    raise IndexError
            elif isinstance(key[0], int) and isinstance(key[1], int):
                return self._grid[key]
        raise IndexError

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], Direction) and isinstance(key[1], int):
                if not isinstance(value, str) or self[key].length != len(value):
                    raise ValueError
                self._grid[self._get_word_mask(self[key].index)] = list(value.upper())
            elif isinstance(key[0], int) and isinstance(key[1], int):
                if not isinstance(value, str) or len(value) != 1:
                    raise ValueError
                needs_parse = self._grid[key] == black or value == black
                self._grid[key] = value.upper()
                if needs_parse:
                    self._parse_grid()
            else:
                raise IndexError
        else:
            raise IndexError

    def __deepcopy__(self, memo):
        copied = copy.copy(self)
        copied._grid = np.copy(copied._grid)
        # Update word references
        copied._words = {
            w.index: Word(copied, w.direction, w.number, w.clue)
            for w in self._words.values()
        }
        return copied

    def __repr__(self):
        return self._text_grid()

    @classmethod
    def from_puz(cls, filename: str) -> "Crossword":
        """Creates a Crossword object from a .puz file.

        Args:
            filename (str): The path of the input .puz file.

        Returns:
            Crossword: A Crossword object.
        """
        puz_obj = puz.read(filename)
        grid = np.reshape(
            list(puz_obj.solution.replace(".", black).replace("-", empty)),
            (puz_obj.height, puz_obj.width),
        )
        xw = cls(grid=grid)
        for cn in puz_obj.clue_numbering().across:
            xw[across, cn["num"]].clue = cn["clue"]
        for cn in puz_obj.clue_numbering().down:
            xw[down, cn["num"]].clue = cn["clue"]
        return xw

    def to_puz(self, filename: str):
        """Ouputs a .puz file from the Crossword object.

        Args:
            filename (str): The output path.
        """
        puz_obj = puz.Puzzle()
        puz_obj.height = self.num_rows
        puz_obj.width = self.num_cols
        puz_obj.solution = (
            "".join(self._grid.ravel()).replace(black, ".").replace(empty, "-")
        )
        fill_grid = self._grid.copy()
        fill_grid[fill_grid != black] = "-"
        fill_grid[fill_grid == black] = "."
        puz_obj.fill = "".join(fill_grid.ravel())
        sorted_words = sorted(
            list(self.iterwords()),
            key=lambda w: w.number + (0.5 if w.direction == down else 0),
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
    ):
        """Ouputs a .pdf file in NYT submission format from the Crossword object.

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
            {clue_rows(across)}
            <tr><td></td></tr>
            <tr><td colspan="3">DOWN</td></tr>
            {clue_rows(down)}
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
    def grid(self) -> np.ndarray:
        """np.ndarray: The raw letter grid as a numpy array"""
        return self._grid

    @property
    def clues(self) -> Dict[Tuple[Direction, int], str]:
        """Dict[Tuple[Direction, int], str]: A dict mapping word index to clue."""
        return {index: w.clue for index, w in self._words.items()}

    def get_symmetric_index(self, index: Tuple[int, int]) -> Tuple[int, int]:
        """Gets the index of a symmetric grid cell. Useful for enforcing symmetry.

        Args:
            index (Tuple[int, int]): The input index.

        Returns:
            Tuple[int, int]: The index of the cell symmetric to the input.
        """
        return (self._num_rows - 1 - index[0], self._num_cols - 1 - index[1])

    def _parse_grid(self):
        """Updates all indices to reflect the state of the _grid property."""
        old_across, old_down = self._across, self._down
        shifted_down = np.pad(self._grid, ((1, 0), (0, 0)), constant_values=black)[
            :-1, :
        ]
        shifted_right = np.pad(self._grid, ((0, 0), (1, 0)), constant_values=black)[
            :, :-1
        ]
        is_open = self._grid != black
        new_num = (is_open) & ((shifted_down == black) | (shifted_right == black))
        self._numbers = (
            np.reshape(np.cumsum(np.ravel(new_num)), self._grid.shape) * new_num
        )
        self._across = np.maximum.accumulate(
            (shifted_right == black) * self._numbers, axis=1
        ) * (is_open)
        self._down = np.maximum.accumulate((shifted_down == black) * self._numbers) * (
            is_open
        )

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
        for across_num in affected_across - {n for d, n in removed if d == across}:
            self._words[(across, int(across_num))] = Word(self, across, int(across_num))
        for down_num in affected_down - {n for d, n in removed if d == down}:
            self._words[(down, int(down_num))] = Word(self, down, int(down_num))

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

    def _get_word_mask(self, word_index: Tuple[Direction, int]) -> np.ndarray:
        """A boolean mask that indicates which grid cells belong to a word.

        Args:
            word_index (Tuple[Direction, int]): The index of the desired word.

        Returns:
            np.ndarray: The grid indicating which cells are in the input word.
        """
        word = self[word_index]
        return self._get_direction_mask(word.direction) == word.number

    def iterwords(self, direction: Optional[Direction] = None) -> Iterator[Word]:
        """Method for iterating over the words in the crossword.

        Args:
            direction (Direction, optional): If provided, limits the iterator to only
                the given direction.

        Yields:
            Iterator[Word]: An iterator of Word objects. Ordered in standard crossword
                fashion (ascending numbers, across then down).
        """
        sorted_words = sorted(
            self._words.values(),
            key=lambda w: w.number
            + (self._num_rows * self._num_cols) * int(w.direction == down),
        )
        for word in sorted_words:
            if direction is None or direction == word.direction:
                yield word

    def get_indices(self, word_index: Tuple[Direction, int]) -> List[Tuple[int, int]]:
        """Gets the list of cell indices for a given word.

        Args:
            word_index (Tuple[Direction, int]): The index of the desired word.

        Returns:
            List[Tuple[int, int]]: A list of cell indices that belong to the word.
        """
        return [
            (int(x[0]), int(x[1]))
            for x in np.argwhere(self._get_word_mask(word_index)).tolist()
        ]

    def get_words_at_index(self, index: Tuple[int, int]) -> Optional[Tuple[Word, Word]]:
        """Gets the two words (across and down) that pass through a cell.

        Args:
            index (Tuple[int, int]): The index of the cell.

        Returns:
            Optional[Tuple[Word, Word]]: A tuple of the (across, down) words at the
                index. If the index corresponds to a black square, this method returns
                None.
        """
        if self[index] != black:
            across_num, down_num = int(self._across[index]), int(self._down[index])
            return self[across, across_num], self[down, down_num]

    def set_word(
        self, word_index: Tuple[Direction, int], value: str, inplace: bool = True
    ) -> Optional["Crossword"]:
        """Sets a word to a new value. If inpace is set to False, returns a new
        Crossword object rather than modifying the current one.

        Args:
            word_index (Tuple[Direction, int]): The index of the word.
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
        self, index: Tuple[int, int], value: str, inplace: bool = True
    ) -> Optional["Crossword"]:
        """Sets a cell to a new value. If inpace is set to False, returns a new
        Crossword object rather than modifying the current one.

        Args:
            index (Tuple[int, int]): The index of the cell.
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
        word_indices: List[Tuple[Direction, int]],
        dictionary: Optional[Dictionary] = None,
        beam_width: int = 100,
    ) -> List["Crossword"]:
        """Finds compatible solutions for a set of words. Uses a beam search algorithm.

        Args:
            word_indices (List[Tuple[Direction, int]]): The indices of the words to
                search for solutions. Only these words are guaranteed to have valid
                solutions in the output.
            dictionary (Union[Dict[str, float], List[str]], optional): A dictionary to
                use as for finding matches. If None, the crossword's default dictionary
                is used.
            beam_width (int): A parameter of the beam search algorithm that controls
                how many solution branches to explore at once. Larger means a more
                exhaustive (slower) search. Defaults to 100.

        Returns:
            List[Crossword]: A list of Crossword objects containing solutions, sorted
            by the product of the solution scores of input words.
        """
        dictionary = parse_dictionary(dictionary) if dictionary else self._dictionary
        sorted_indices = sorted(
            word_indices, key=lambda i: len(self[i].find_matches(dictionary=dictionary))
        )
        memory = [self]
        for word_index in tqdm(sorted_indices):
            new_memory = []
            for xw in memory:
                scored_matches = xw[word_index].find_matches(
                    sort_method="cross_match", dictionary=dictionary, return_scores=True
                )
                new_memory += [
                    (xw.set_word(word_index, match, inplace=False), score)
                    for match, score in scored_matches[:beam_width]
                ]
            memory = [
                k for k, v in sorted(new_memory, key=lambda x: x[1], reverse=True)
            ][:beam_width]
        scored_memory = [
            (grid, np.product([dictionary[grid[idx].value] for idx in sorted_indices]))
            for grid in memory
        ]
        return [
            xw for xw, score in sorted(scored_memory, key=lambda x: x[1], reverse=True)
        ]

    def find_area_solutions(
        self,
        seed_word_index: Tuple[Direction, int],
        dictionary: Optional[Dictionary] = None,
        beam_width: int = 100,
    ) -> List["Crossword"]:
        """Finds compatible solutions for a contiguous unfilled area from a seed word.
        Uses a beam search algorithm.

        Args:
            seed_word_index (Tuple[Direction, int]): The starting word. The search will
                expand to include all contiguous open words including crosses.
            dictionary (Union[Dict[str, float], List[str]], optional): A dictionary to
                use as for finding matches. If None, the crossword's default dictionary
                is used.
            beam_width (int): A parameter of the beam search algorithm that controls
                how many solution branches to explore at once. Larger means a more
                exhaustive (slower) search. Defaults to 100.

        Returns:
            List[Crossword]: A list of Crossword objects containing solutions, sorted
                by the product of the solution scores of all words included in the
                contiguous area.
        """
        word_index_set = {seed_word_index}
        while True:
            new_word_index_set = copy.deepcopy(word_index_set)
            for word_index in word_index_set:
                new_word_index_set |= {
                    cross.index
                    for letter, cross in zip(
                        self[word_index].value, self[word_index].get_crosses()
                    )
                    if cross.is_open() and letter == empty
                }
            if new_word_index_set == word_index_set:
                break
            word_index_set = new_word_index_set
        solutions = self.find_solutions(list(word_index_set), dictionary, beam_width)
        dictionary = parse_dictionary(dictionary) if dictionary else self._dictionary
        scores = [
            np.product(
                [
                    dictionary[solution[word_index].value]
                    for word_index in word_index_set
                ]
            )
            for solution in solutions
        ]

        return [
            solution
            for solution, score in sorted(
                zip(solutions, scores), key=lambda x: x[1], reverse=True
            )
        ]

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
                if self._grid[i, j] == black:
                    value = "███"
                elif not numbers:
                    value = f" {self._grid[i,j]} "
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
                    f"""<td class='xw{suffix}{ f" black{suffix}" if self._grid[r, c] == black else ""}'>
                        <div class='number{suffix}'> {number if number else ""}</div>
                        <div class='value{suffix}'>{self._grid[r,c] if self._grid[r,c] != black else ""}</div>
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
