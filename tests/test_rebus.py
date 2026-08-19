import os
import tempfile

import pytest

from blacksquare import ACROSS, DOWN, EMPTY, Crossword, Rebus
from blacksquare.types import SpecialCellValue


class TestRebusObject:
    def test_symmetric_rebus_creation(self):
        r1 = Rebus("FOO")
        assert r1.across == "FOO"
        assert r1.down == "FOO"
        assert r1.value == "FOO"
        assert r1.is_symmetric is True
        assert r1.get_value(ACROSS) == "FOO"
        assert r1.get_value(DOWN) == "FOO"
        assert str(r1) == "FOO"
        assert repr(r1) == "Rebus('FOO')"

    def test_asymmetric_rebus_creation(self):
        r = Rebus("FOO", "BAR")
        assert r.across == "FOO"
        assert r.down == "BAR"
        assert r.value == ("FOO", "BAR")
        assert r.is_symmetric is False
        assert r.get_value(ACROSS) == "FOO"
        assert r.get_value(DOWN) == "BAR"
        assert str(r) == "FOO/BAR"
        assert repr(r) == "Rebus(across='FOO', down='BAR')"

    def test_keyword_arguments(self):
        r1 = Rebus(across="CAT", down="DOG")
        assert r1.across == "CAT"
        assert r1.down == "DOG"

        r2 = Rebus(value="HEART")
        assert r2.across == "HEART"
        assert r2.down == "HEART"

        r3 = Rebus(across="BIRD")
        assert r3.across == "BIRD"
        assert r3.down == "BIRD"

        r4 = Rebus(down="FISH")
        assert r4.across == "FISH"
        assert r4.down == "FISH"

    def test_normalization_and_stripping(self):
        r = Rebus("  foo  ", "  bar  ")
        assert r.across == "FOO"
        assert r.down == "BAR"

    def test_equality_and_hashing(self):
        r1 = Rebus("FOO")
        r2 = Rebus("foo")
        r3 = Rebus("FOO", "BAR")
        r4 = Rebus("FOO", "BAR")

        assert r1 == r2
        assert r1 == "FOO"
        assert r1 == "foo"
        assert r3 == r4
        assert r1 != r3
        assert r3 != "FOO"
        assert r3 != 42

        rebus_set = {r1, r2, r3, r4}
        assert len(rebus_set) == 2

    def test_invalid_rebus_inputs(self):
        with pytest.raises(ValueError):
            Rebus()

        with pytest.raises(ValueError):
            Rebus("")

        with pytest.raises(ValueError):
            Rebus("   ")

        with pytest.raises(ValueError):
            Rebus("FOO", "")

        with pytest.raises(ValueError):
            Rebus(123)  # type: ignore

        with pytest.raises(ValueError):
            Rebus("FOO", across="BAR")


class TestCrosswordRebusCells:
    def test_set_symmetric_rebus_cell(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("FOO")

        cell = xw[0, 2]
        assert isinstance(cell.value, Rebus)
        assert cell.value == Rebus("FOO")
        assert cell.str == "FOO"
        assert repr(cell) == "Cell(Rebus('FOO'))"
        assert not cell.is_open()

    def test_set_asymmetric_rebus_cell(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[1, 1] = Rebus(across="HEART", down="LOVE")

        cell = xw[1, 1]
        assert isinstance(cell.value, Rebus)
        assert cell.value.across == "HEART"
        assert cell.value.down == "LOVE"
        assert cell.str == "HEART/LOVE"
        assert repr(cell) == "Cell(Rebus(across='HEART', down='LOVE'))"

    def test_overwrite_rebus_cell(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 0] = Rebus("TEST")
        assert xw[0, 0].value == Rebus("TEST")

        xw[0, 0] = "A"
        assert xw[0, 0].value == "A"
        assert xw[0, 0].str == "A"

        xw[0, 0] = EMPTY
        assert xw[0, 0].value == SpecialCellValue.EMPTY
        assert xw[0, 0].is_open()


class TestWordRebusRenderingAndParsing:
    def test_word_value_rendering(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("FOO")
        xw[0, 0] = "A"
        xw[0, 1] = "B"
        xw[0, 3] = "C"
        xw[0, 4] = "D"

        word = xw[ACROSS, 1]
        assert word.value == "AB(FOO)CD"
        assert repr(word) == 'Word(Across 1: "AB(FOO)CD")'
        assert len(word) == 5

    def test_word_value_rendering_asymmetric(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus(across="HEART", down="LOVE")
        xw[0, 0] = "A"
        xw[0, 1] = "B"
        xw[0, 3] = "C"
        xw[0, 4] = "D"

        xw[1, 2] = "X"
        xw[2, 2] = "Y"
        xw[3, 2] = "Z"
        xw[4, 2] = "W"

        across_word = xw[ACROSS, 1]
        down_word = xw[DOWN, 3]

        assert across_word.value == "AB(HEART)CD"
        assert down_word.value == "(LOVE)XYZW"
        assert repr(across_word) == 'Word(Across 1: "AB(HEART)CD")'
        assert repr(down_word) == 'Word(Down 3: "(LOVE)XYZW")'

    def test_setting_word_with_plain_rebus_string(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("FOO")

        # The across word has 5 cells: 1 + 1 + 3 (rebus) + 1 + 1 = 7 chars
        xw[ACROSS, 1] = "ABFOOCD"
        assert xw[0, 0].value == "A"
        assert xw[0, 1].value == "B"
        assert xw[0, 2].value == Rebus("FOO")
        assert xw[0, 3].value == "C"
        assert xw[0, 4].value == "D"
        assert xw[ACROSS, 1].value == "AB(FOO)CD"

    def test_setting_word_with_asymmetric_rebus_preservation(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus(across="OLD", down="LOVE")

        # Update across word using plain string with across length 5:
        xw[ACROSS, 1] = "ABHEARTCD"
        assert xw[0, 2].value == Rebus(across="HEART", down="LOVE")
        assert xw[ACROSS, 1].value == "AB(HEART)CD"
        assert xw[DOWN, 3].value == "(LOVE)    "

    def test_setting_word_with_explicit_parentheses(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "AB(FOO)CD"

        assert xw[0, 0].value == "A"
        assert xw[0, 1].value == "B"
        assert xw[0, 2].value == Rebus("FOO")
        assert xw[0, 3].value == "C"
        assert xw[0, 4].value == "D"
        assert xw[ACROSS, 1].value == "AB(FOO)CD"
        assert xw[DOWN, 3].value == "(FOO)    "

    def test_setting_word_with_multiple_rebuses(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "A(CAT)B(DOG)C"

        assert xw[0, 0].value == "A"
        assert xw[0, 1].value == Rebus("CAT")
        assert xw[0, 2].value == "B"
        assert xw[0, 3].value == Rebus("DOG")
        assert xw[0, 4].value == "C"
        assert xw[ACROSS, 1].value == "A(CAT)B(DOG)C"

        # Now set using plain concatenated characters:
        xw[ACROSS, 1] = "XCATYDOGZ"
        assert xw[0, 0].value == "X"
        assert xw[0, 1].value == Rebus("CAT")
        assert xw[0, 2].value == "Y"
        assert xw[0, 3].value == Rebus("DOG")
        assert xw[0, 4].value == "Z"
        assert xw[ACROSS, 1].value == "X(CAT)Y(DOG)Z"


class TestRebusIntegrationAndExport:
    def test_copy_crossword_with_rebus(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus(across="SUN", down="MOON")
        xw[ACROSS, 1] = "AB(SUN)CD"

        copied = xw.copy()
        assert copied[0, 2].value == Rebus(across="SUN", down="MOON")
        assert copied[ACROSS, 1].value == "AB(SUN)CD"
        assert copied[DOWN, 3].value == "(MOON)    "

        # Ensure deep independence
        copied[0, 2] = "Z"
        assert copied[0, 2].value == "Z"
        assert xw[0, 2].value == Rebus(across="SUN", down="MOON")

    def test_puz_export_with_rebus(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("FOO")
        xw[ACROSS, 1] = "AB(FOO)CD"

        with tempfile.NamedTemporaryFile(suffix=".puz", delete=False) as f:
            puz_path = f.name

        try:
            xw.to_puz(puz_path)
            assert os.path.exists(puz_path)
            assert os.path.getsize(puz_path) > 0
        finally:
            if os.path.exists(puz_path):
                os.remove(puz_path)

    def test_pdf_export_with_rebus(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("HEART")
        xw[ACROSS, 1] = "AB(HEART)CD"
        xw[ACROSS, 1].clue = "Love song symbol"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            xw.to_pdf(pdf_path, header=["Author", "Test"])
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
