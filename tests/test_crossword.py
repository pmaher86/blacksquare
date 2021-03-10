from blacksquare.crossword import Crossword, Word, across, down, black, empty
import pytest
import numpy as np
from pathlib import Path
from blacksquare.dictionary import parse_dictionary


@pytest.fixture
def simple_dict():
    return {"AA": 0.9, "AB": 1.0, "BB": 0.01, "BC": 0.1, "ABC": 1.0, "BBB": 0.01}


@pytest.fixture
def xw(simple_dict):
    """
    ┌───┬───┬───┐   ┌───┬───┬───┐
    │███│ B │ C │   │███│1  │2  │
    ├───┼───┼───┤   ├───┼───┼───┤
    │ A │ B │ C │   │3  │   │   │
    ├───┼───┼───┤   ├───┼───┼───┤
    │   │   │███│   │4  │   │███│
    └───┴───┴───┘   └───┴───┴───┘
    """
    xw = Crossword(3, dictionary=simple_dict)
    xw[0, 0] = black
    xw[2, 2] = black
    xw[across, 1] = "BC"
    xw[across, 3] = "ABC"
    return xw


class TestCrosswordIndexing:
    def test_index_word(self, xw):
        assert xw[across, 1].value == "BC"
        assert xw[across, 4].value == "  "
        assert xw[down, 1].value == "BB "

    def test_index_letter(self, xw):
        assert xw[0, 0] == black
        assert xw[1, 0] == "A"
        assert xw[2, 1] == " "
        assert xw[1, -1] == "C"

    def test_index_setter_word(self, xw):
        xw[across, 4] = "qZ"
        assert xw[across, 4].value == "QZ"
        assert xw[down, 1].value == "BBZ"

    def test_index_setter_letter(self, xw):
        xw[2, 1] = "z"
        assert xw[across, 4].value == " Z"
        assert xw[down, 1].value == "BBZ"

    def test_wrong_word_length(self, xw):
        for bad_val in ["A", "AAA"]:
            with pytest.raises(ValueError):
                xw[across, 4] = bad_val
        with pytest.raises(ValueError):
            xw[1, 0] = "AA"

    @pytest.mark.parametrize("bad_val", [["A", "A"], 72, Crossword(2)[across, 1]])
    def test_wrong_types(self, xw, bad_val):
        with pytest.raises(ValueError):
            xw[across, 4] = bad_val

    @pytest.mark.parametrize(
        "bad_index", [(across, 2), down, (1, 1, 0), (1, slice(1, 2))]
    )
    def test_invalid_index(self, xw, bad_index):
        with pytest.raises(IndexError):
            xw[bad_index]
        with pytest.raises(IndexError):
            xw[bad_index] = "A"


class TestWordIndexing:
    def test_getter(self, xw):
        assert xw[across, 1][0] == "B"
        assert xw[across, 1][-1] == "C"

    def test_setter(self, xw):
        xw[across, 1][0] = "q"
        assert xw[across, 1].value == "QC"

    @pytest.mark.parametrize("bad_index", [9, "A", slice(2)])
    def test_out_of_bounds_index(self, xw, bad_index):
        with pytest.raises(IndexError):
            xw[across, 3][bad_index]
        with pytest.raises(IndexError):
            xw[across, 3][bad_index] = "Q"

    @pytest.mark.parametrize("bad_value", ["", "AB", 4])
    def test_invalid_value(self, xw, bad_value):
        with pytest.raises(ValueError):
            xw[across, 3][0] = bad_value


class TestParsing:
    def test_parsing_indices(self):
        xw = Crossword(5)
        black_inds = [(0, 0), (0, 1), (2, 2), (4, 4), (4, 3)]
        for ind in black_inds:
            xw[ind] = black
        expected_keys = [
            (across, 1),
            (across, 4),
            (across, 6),
            (across, 7),
            (across, 8),
            (across, 10),
            (down, 1),
            (down, 2),
            (down, 3),
            (down, 4),
            (down, 5),
            (down, 9),
        ]
        assert set(expected_keys) == set([w.index for w in xw.iterwords()])

    def test_reparse_after_removing_black(self, xw):
        old_indices = set([w.index for w in xw.iterwords()])
        xw[0, 0] = " "
        assert set([w.index for w in xw.iterwords()]) != old_indices

    def test_clues_after_reparse(self, xw):
        xw[across, 4].clue = "clue to forget"
        xw[down, 1].clue = "clue to keep"
        xw[2, 2] = empty
        assert xw[across, 4].clue == ""
        assert xw[down, 1].clue == "clue to keep"


class TestConversion:
    def test_to_puz(self, xw, tmp_path):
        filename = tmp_path / "test.puz"
        xw.to_puz(filename)
        assert tmp_path.exists()
        loaded = Crossword.from_puz(filename)
        assert np.all(loaded.grid == xw.grid)

    def test_from_puz(self):
        Crossword.from_puz(Path(__file__).parent / "dummy.puz")

    def test_to_pdf(self, xw, tmp_path):
        filename = tmp_path / "test.pdf"
        xw.to_pdf(filename, ["Line 1", "Line 2"])
        assert tmp_path.exists()


class TestFindMatches:
    def test_find_matches(self, xw, simple_dict):
        empty_matches = xw[across, 4].find_matches()
        assert set(empty_matches) == {k for k in simple_dict.keys() if len(k) == 2}
        assert set(xw[down, 3].find_matches()) == {"AA", "AB"}
        assert xw[across, 3].find_matches() == ["ABC"]
        assert xw[down, 2].find_matches() == []
        xw[1, 0] = "Z"
        assert xw[down, 3].find_matches() == []

    def test_find_matches_alpha(self, xw):
        matches = xw[across, 4].find_matches(sort_method="alpha")
        assert matches == ["AA", "AB", "BB", "BC"]

    def test_find_matches_score(self, xw):
        matches = xw[across, 4].find_matches(sort_method="score")
        assert matches == ["AB", "AA", "BC", "BB"]
        matches = xw[across, 4].find_matches(sort_method="score", return_scores=True)
        assert matches == [("AB", 1.0), ("AA", 0.9), ("BC", 0.1), ("BB", 0.01)]

    def test_find_matches_cross(self, xw):
        matches = xw[down, 3].find_matches(sort_method="cross_match")
        assert matches == ["AA", "AB"]
        matches = xw[down, 3].find_matches(
            sort_method="cross_match", return_scores=True
        )
        assert matches == [("AA", 0.9 * (1.0 + 0.9)), ("AB", 1.0 * (0.01 + 0.1))]

    def test_find_matches_cross_no_matches(self, xw):
        assert xw[down, 2].find_matches(sort_method="cross_match") == []
        xw[1, 0] = "Z"
        assert xw[down, 3].find_matches(sort_method="cross_match") == []

    def test_find_matches_custom_dict(self, xw):
        assert xw[down, 3].find_matches(dictionary=["AZ", "ZZ"]) == ["AZ"]


class TestWordMethods:
    def test_is_open(self, xw):
        assert not xw[across, 1].is_open()
        assert xw[across, 4].is_open()
        assert xw[down, 1].is_open()

    def test_length(self, xw):
        assert xw[across, 1].length == 2
        assert xw[across, 3].length == 3

    def test_get_crosses(self, xw):
        crosses = xw[across, 3].get_crosses()
        assert {c.index for c in crosses} == {(down, 1), (down, 2), (down, 3)}


class TestCrosswordProperties:
    def test_symmetric_index(self):
        xw = Crossword(5, 7)
        assert xw.get_symmetric_index((0, 1)) == (4, 5)
        assert xw.get_symmetric_index((2, 3)) == (2, 3)

    def test_num_rows_cols(self):
        xw = Crossword(5, 7)
        assert xw.num_rows == 5
        assert xw.num_cols == 7

    def test_grid(self, xw):
        assert np.all(xw.grid == [list("#BC"), list("ABC"), list("  #")])

    def test_iterwords(self, xw):
        across_indices = [(across, n) for n in [1, 3, 4]]
        down_indices = [(down, n) for n in [1, 2, 3]]
        assert list(w.index for w in xw.iterwords()) == across_indices + down_indices
        assert list(w.index for w in xw.iterwords(across)) == across_indices
        assert list(w.index for w in xw.iterwords(down)) == down_indices

    def test_get_indices(self, xw):
        assert xw.get_indices((across, 3)) == [(1, 0), (1, 1), (1, 2)]

    def test_get_words_at_index(self, xw):
        assert xw.get_words_at_index((1, 1)) == (xw[across, 3], xw[down, 1])
        assert xw.get_words_at_index((0, 0)) is None

    def test_set_word(self, xw):
        xw.set_word((across, 1), "DD")
        assert xw[across, 1].value == "DD"
        xw2 = xw.set_word((across, 4), "EE", inplace=False)
        assert xw2 is not xw
        assert xw2[across, 4].value == "EE"
        assert xw[across, 4].value == "  "

    def test_set_cell(self, xw):
        xw.set_cell((0, 1), "D")
        assert xw[across, 1].value == "DC"
        xw2 = xw.set_cell((2, 1), "E", inplace=False)
        assert xw2 is not xw
        assert xw2[across, 4].value == " E"
        assert xw[across, 4].value == "  "


class TestCrosswordSolutions:
    def test_find_solutions(self, xw):
        solutions = xw.find_solutions([(down, 1), (down, 3)])
        assert len(solutions) == 2
        assert solutions[0][across, 4].value == "BB"

    def test_find_solutions_with_custom_dictionary(self, xw):
        solutions = xw.find_solutions(
            [(down, 1), (down, 3)],
            dictionary={"BBB": 1, "AC": 1, "CB": 1, "AA": 0.1, "AB": 0.1},
        )
        assert len(solutions) == 3
        assert solutions[0][across, 4].value == "CB"

    def test_find_solutions_on_solved_words(self, xw):
        solutions = xw.find_solutions([(across, 1), (across, 3)])
        assert len(solutions) == 1
        solutions = xw.find_solutions([(across, 1), (down, 2)])
        assert len(solutions) == 0

    def test_find_area_solutions(self, xw):
        xw[1, 2] = empty
        solutions = xw.find_area_solutions((down, 3))
        assert len(solutions) == 2
        assert solutions[0][across, 4].value == "AB"
        assert all(s[1, 2] == empty for s in solutions)

    def test_find_area_solutions_on_solved(self, xw):
        solutions = xw.find_area_solutions((across, 1))
        assert len(solutions) == 1
        assert np.all(solutions[0].grid == xw.grid)

    def test_unsolvable(self, xw):
        xw[1, 0] = "J"
        xw[1, 1] = empty
        xw[0, 1] = "Q"
        solutions = xw.find_area_solutions((down, 1))
        assert solutions == []