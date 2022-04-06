import pytest

from blacksquare import ACROSS, BLACK, Crossword, WordList


@pytest.fixture
def word_list() -> WordList:
    return WordList(
        {
            "AA": 0.01,
            "AB": 0.5,
            "BB": 0.5,
            "ABC": 1.0,
            "BCD": 0.1,
            "BBB": 0.1,
            "ABB": 1.0,
            "CCB": 1.0,
            "BBCD": 1.0,
        }
    )


@pytest.fixture
def xw(word_list: WordList) -> Crossword:
    """
    ┌───┬───┬───┬───┐
    │███│ B │ C │ D │
    ├───┼───┼───┼───┤
    │ A │ B │ C │ D │
    ├───┼───┼───┼───┤
    │   │   │   │███│
    └───┴───┴───┴───┘
    ┌───┬───┬───┬───┐
    │███│1  │2  │3  │
    ├───┼───┼───┼───┤
    │4  │   │   │   │
    ├───┼───┼───┼───┤
    │5  │   │   │███│
    └───┴───┴───┴───┘
    """
    xw = Crossword(3, 4, word_list=word_list)
    xw[0, 0] = BLACK
    xw[ACROSS, 1] = "BCD"
    xw[ACROSS, 4] = "ABCD"
    return xw
