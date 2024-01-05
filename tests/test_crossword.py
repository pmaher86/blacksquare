from pathlib import Path

import numpy as np
import pytest

from blacksquare import ACROSS, BLACK, DOWN, EMPTY, Crossword, WordList
from blacksquare.symmetry import Symmetry


class TestCrosswordIndexing:
    def test_index_setter_word(self, xw):
        xw[ACROSS, 5] = "qZb"
        assert xw[ACROSS, 5].value == "QZB"
        assert xw[DOWN, 1].value == "BBZ"

    def test_index_setter_letter(self, xw):
        xw[2, 1] = "z"
        assert xw[ACROSS, 5].value == " Z "
        assert xw[DOWN, 1].value == "BBZ"

    def test_wrong_word_length(self, xw):
        for bad_val in ["AA", "AAAA"]:
            with pytest.raises(ValueError):
                xw[ACROSS, 5] = bad_val
        with pytest.raises(ValueError):
            xw[1, 0] = "AA"

    @pytest.mark.parametrize("bad_val", [["A", "A"], 72, Crossword(2)[ACROSS, 1]])
    def test_wrong_types(self, xw, bad_val):
        with pytest.raises(ValueError):
            xw[ACROSS, 5] = bad_val

    @pytest.mark.parametrize(
        "bad_index", [(ACROSS, 2), DOWN, (1, 1, 0), (1, slice(1, 2))]
    )
    def test_invalid_index(self, xw, bad_index):
        with pytest.raises(IndexError):
            xw[bad_index]
        with pytest.raises(IndexError):
            xw[bad_index] = "A"

    def test_unchecked_word_indexing(self):
        xw = Crossword(5, 5, symmetry=None)
        ind = [1, 2, 3]
        for i, j in zip(ind, ind[::-1]):
            xw[i, j] = BLACK
        xw[DOWN, 1] = "ABCDE"
        xw[DOWN, 1] = "     "
        xw.fill()


class TestParsing:
    def test_parsing_indices(self):
        xw = Crossword(5)
        BLACK_inds = [(0, 0), (0, 1), (2, 2), (4, 4), (4, 3)]
        for ind in BLACK_inds:
            xw[ind] = BLACK
        expected_keys = [
            (ACROSS, 1),
            (ACROSS, 4),
            (ACROSS, 6),
            (ACROSS, 7),
            (ACROSS, 8),
            (ACROSS, 10),
            (DOWN, 1),
            (DOWN, 2),
            (DOWN, 3),
            (DOWN, 4),
            (DOWN, 5),
            (DOWN, 9),
        ]
        assert set(expected_keys) == set([w.index for w in xw.iterwords()])

    def test_reparse_after_removing_black(self, xw):
        old_indices = set([w.index for w in xw.iterwords()])
        xw[0, 0] = " "
        assert set([w.index for w in xw.iterwords()]) != old_indices

    def test_clues_after_reparse(self, xw: Crossword):
        xw[ACROSS, 5].clue = "across to forget"
        xw[ACROSS, 4].clue = "across to keep"
        xw[DOWN, 4].clue = "down to keep"
        xw[DOWN, 2].clue = "down to forget"
        xw[2, 2] = BLACK
        assert xw[ACROSS, 3].clue == "across to keep"
        assert xw[DOWN, 3].clue == "down to keep"
        for k, c in xw.clues.items():
            if k not in ((ACROSS, 3), (DOWN, 3)):
                assert c == ""

    def test_single_letter(self):
        grid = [
            ["#", "#", "#", "#", "A"],
            ["B", "A", "D", "#", "C"],
            ["#", "#", "#", "#", "T"],
        ]
        xw = Crossword(grid=grid)
        assert xw[ACROSS, 2].value == "BAD"


class TestConversion:
    def test_to_puz(self, xw, tmp_path):
        filename = tmp_path / "test.puz"
        xw.to_puz(filename)
        assert tmp_path.exists()
        loaded = Crossword.from_puz(filename)
        assert all(l == x for l, x in zip(loaded.itercells(), xw.itercells()))

    def test_from_puz(self):
        Crossword.from_puz(Path(__file__).parent / "dummy.puz")

    def test_to_pdf(self, xw, tmp_path):
        filename = tmp_path / "test.pdf"
        xw.to_pdf(filename, ["Line 1", "Line 2"])
        assert tmp_path.exists()


class TestCrosswordProperties:
    def test_symmetric_cell_index(self):
        xw = Crossword(5, 7)
        assert xw.get_symmetric_cell_index((0, 1)) == (4, 5)
        assert xw.get_symmetric_cell_index((2, 3)) == (2, 3)

    def test_num_rows_cols(self):
        xw = Crossword(5, 7)
        assert xw.num_rows == 5
        assert xw.num_cols == 7

    def test_iterwords(self, xw):
        across_indices = [(ACROSS, n) for n in [1, 4, 5]]
        down_indices = [(DOWN, n) for n in [1, 2, 3, 4]]
        assert list(w.index for w in xw.iterwords()) == across_indices + down_indices
        assert list(w.index for w in xw.iterwords(ACROSS)) == across_indices
        assert list(w.index for w in xw.iterwords(DOWN)) == down_indices

    def test_get_indices(self, xw):
        assert xw.get_indices((ACROSS, 4)) == [(1, 0), (1, 1), (1, 2), (1, 3)]

    def test_get_word_at_index(self, xw: Crossword):
        assert xw.get_word_at_index((1, 1), ACROSS) == xw[ACROSS, 4]
        assert xw.get_word_at_index((1, 1), DOWN) == xw[DOWN, 1]
        assert xw.get_word_at_index((0, 0), ACROSS) is None

    def test_set_word(self, xw):
        xw.set_word((ACROSS, 1), "DDD")
        assert xw[ACROSS, 1].value == "DDD"

    def test_set_cell(self, xw):
        xw.set_cell((0, 1), "D")
        assert xw[ACROSS, 1].value == "DCD"

    def test_copy(self, xw):
        xw2 = xw.copy()
        assert xw2 is not xw
        xw2.set_cell((2, 1), "E")
        assert xw2[ACROSS, 5].value == " E "
        assert xw[ACROSS, 5].value == "   "

    def test_subgrids(self, xw: Crossword):
        graphs = xw.get_disconnected_open_subgrids()
        assert len(graphs) == 1
        assert len(graphs[0]) == 4
        xw[2, 0] = "Z"
        xw[1, 3] = " "
        graphs = xw.get_disconnected_open_subgrids()
        assert len(graphs) == 2


class TestCrosswordFill:
    def test_fill(self, xw):
        solution = xw.fill()
        assert solution[ACROSS, 5].value == "BBB"

    def test_find_solutions_with_custom_dictionary(self, xw):
        solution = xw.fill(
            word_list=WordList(
                {
                    "BBB": 1,
                    "AC": 1,
                    "CBX": 1,
                    "CBA": 1,
                    "AA": 0.1,
                    "AB": 0.1,
                    "CCX": 1,
                    "CCA": 0.1,
                }
            ),
        )
        assert solution[ACROSS, 5].value == "CBX"

    def test_unsolvable(self, xw):
        xw[1, 0] = "J"
        xw[1, 1] = EMPTY
        xw[0, 1] = "Q"
        solution = xw.fill()
        assert solution is None


def test_symmetry_requirements():
    with pytest.raises(ValueError):
        Crossword(5, 7, symmetry=Symmetry.FULL)
