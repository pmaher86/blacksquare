import pytest

from blacksquare import ACROSS, BLACK, DOWN, EMPTY


class TestGetParentWord:
    def test_get_parent_word(self, xw):
        assert xw[1, 1].get_parent_word(ACROSS) is xw[ACROSS, 4]
        assert xw[1, 1].get_parent_word(DOWN) is xw[DOWN, 1]

    def test_black_cell(self, xw):
        assert xw[0, 0].get_parent_word(ACROSS) is None


class TestSetter:
    def test_empty(self, xw):
        xw[0, 1].value = EMPTY
        xw[ACROSS, 1].value == " CD"

    def test_black(self, xw):
        xw[0, 1].value = BLACK
        xw[0, 2].get_parent_word(DOWN).index == (DOWN, 1)

    def test_lowercase(self, xw):
        xw[0, 1].value = "a"
        xw[ACROSS, 1].value == "ACD"

    def test_invalid(self, xw):
        with pytest.raises(ValueError):
            xw[0, 1].value = "AB"


class TestStr:
    def test_letter(self, xw):
        xw[0, 1].str == "A"

    def test_black(self, xw):
        xw[0, 0].str is None

    def test_empty(self, xw):
        xw[2, 0].str == " "


def test_equality(xw):
    assert xw[0, 0] == BLACK
    assert xw[1, 0] == "A"
    assert xw[2, 1] == EMPTY
    assert xw[1, -1] == "D"
    assert xw[0, 0] != "A"
    assert xw[1, 0] != BLACK


def test_number(xw):
    assert xw[1, 0].number == 4
    assert xw[0, 0].number is None
    assert xw[1, 1].number is None


def test_symmetric_image(xw):
    assert xw[0, 0].symmetric_image.index == (2, 3)
