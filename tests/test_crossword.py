from pathlib import Path

import numpy as np
import pytest

from blacksquare import ACROSS, BLACK, DOWN, EMPTY, Crossword, WordList


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

    def test_clues_after_reparse(self, xw):
        xw[ACROSS, 5].clue = "clue to forget"
        xw[DOWN, 1].clue = "clue to keep"
        xw[2, 3] = EMPTY
        assert xw[ACROSS, 5].clue == ""
        assert xw[DOWN, 1].clue == "clue to keep"


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
    def test_symmetric_index(self):
        xw = Crossword(5, 7)
        assert xw.get_symmetric_index((0, 1)) == (4, 5)
        assert xw.get_symmetric_index((2, 3)) == (2, 3)

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
        xw2 = xw.set_word((ACROSS, 5), "EEE", inplace=False)
        assert xw2 is not xw
        assert xw2[ACROSS, 5].value == "EEE"
        assert xw[ACROSS, 5].value == "   "

    def test_set_cell(self, xw):
        xw.set_cell((0, 1), "D")
        assert xw[ACROSS, 1].value == "DCD"
        xw2 = xw.set_cell((2, 1), "E", inplace=False)
        assert xw2 is not xw
        assert xw2[ACROSS, 5].value == " E "
        assert xw[ACROSS, 5].value == "   "


class TestCrosswordSolutions:
    def test_find_solutions(self, xw):
        solutions = xw.find_solutions([(DOWN, 1), (DOWN, 4)])
        assert len(solutions) == 2
        assert solutions[0][ACROSS, 5].value == "BB "

    def test_find_solutions_with_custom_dictionary(self, xw):
        solutions = xw.find_solutions(
            [(DOWN, 1), (DOWN, 4)],
            word_list=WordList({"BBB": 1, "AC": 1, "CBX": 1, "AA": 0.1, "AB": 0.1}),
        )
        assert len(solutions) == 2
        assert solutions[0][ACROSS, 5].value == "CB "

    def test_find_solutions_on_solved_words(self, xw):
        solutions = xw.find_solutions([(ACROSS, 1), (ACROSS, 4)])
        assert len(solutions) == 0
        solutions = xw.find_solutions([(ACROSS, 4), (DOWN, 2)])
        assert len(solutions) == 0

    def test_find_area_solutions(self, xw):
        xw[1, 0] = EMPTY
        xw[0, 3] = EMPTY
        solutions = xw.find_area_solutions((DOWN, 4))
        assert len(solutions) == 1
        assert solutions[0][ACROSS, 5].value == "BBB"
        assert all(s[0, 3] == EMPTY for s in solutions)

    def test_find_area_solutions_on_solved(self, xw):
        solutions = xw.find_area_solutions((ACROSS, 1))
        assert len(solutions) == 1
        assert np.all(
            [s == x for s, x in zip(solutions[0].itercells(), xw.itercells())]
        )

    def test_unsolvable(self, xw):
        xw[1, 0] = "J"
        xw[1, 1] = EMPTY
        xw[0, 1] = "Q"
        solutions = xw.find_area_solutions((DOWN, 1))
        assert solutions == []
