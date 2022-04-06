import pytest

from blacksquare import ACROSS, BLACK, DOWN, EMPTY, Crossword, Symmetry


class TestFindMatches:
    def test_find_matches(self, xw: Crossword):
        matches = xw[ACROSS, 5].find_matches()
        assert matches.words == ["BBB", "ABB"]

    def test_no_matches(self, xw: Crossword):
        xw[2, 1] = "X"
        matches = xw[ACROSS, 5].find_matches()
        assert len(matches) == 0

    def test_matches_on_complete(self, xw: Crossword):
        xw[0, 1] = "X"
        matches = xw[ACROSS, 1].find_matches()
        assert len(matches) == 0

    def test_matches_on_no_wors(self, xw: Crossword):
        matches = xw[ACROSS, 4].find_matches()
        assert len(matches) == 0


class TestSetter:
    def test_value_setter(self, xw):
        xw[ACROSS, 5] = "abc"
        assert xw[ACROSS, 5].value == "ABC"

    def test_spaces(self, xw):
        xw[ACROSS, 5] = "BC_"
        assert xw[2, 0] == "B"
        assert xw[2, 1] == "C"
        assert xw[2, 2] == EMPTY

    @pytest.mark.parametrize("bad_value", ["AB", ["A", "B", BLACK]])
    def test_invalid_values(self, xw, bad_value):
        with pytest.raises(ValueError):
            xw[ACROSS, 5] = bad_value


def test_is_open(xw):
    assert xw[DOWN, 1].is_open()
    assert not xw[ACROSS, 1].is_open()


def test_length(xw):
    assert len(xw[ACROSS, 1]) == 3
    assert len(xw[ACROSS, 4]) == 4


def test_get_crosses(xw):
    crosses = xw[ACROSS, 4].crosses
    assert [c.index for c in crosses] == [(DOWN, 4), (DOWN, 1), (DOWN, 2), (DOWN, 3)]


class TestWordIndexing:
    def test_getter(self, xw):
        assert xw[ACROSS, 1][0] == "B"
        assert xw[ACROSS, 1][-1] == "D"

    def test_setter(self, xw):
        xw[ACROSS, 1][0] = "q"
        assert xw[ACROSS, 1].value == "QCD"

    @pytest.mark.parametrize("bad_index", [9, "A", slice(2)])
    def test_out_of_bounds_index(self, xw, bad_index):
        with pytest.raises(IndexError):
            xw[ACROSS, 3][bad_index]
        with pytest.raises(IndexError):
            xw[ACROSS, 3][bad_index] = "Q"

    @pytest.mark.parametrize("bad_value", ["", "AB", 4])
    def test_invalid_value(self, xw, bad_value):
        with pytest.raises(ValueError):
            xw[ACROSS, 4][0] = bad_value


def test_symmetric_image(xw):
    assert xw[DOWN, 3].symmetric_image.index == (DOWN, 4)

    diag_xw = Crossword(5, symmetry=Symmetry.NW_DIAGONAL)
    assert diag_xw[ACROSS, 6].symmetric_image.index == (DOWN, 2)
