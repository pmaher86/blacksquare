import io
import os
import tempfile

import pytest

from blacksquare import ACROSS, BLACK, DOWN, EMPTY, Crossword, Rebus
from blacksquare.puz import (
    ExtensionCode,
    GridMarkup,
    PuzData,
    PuzzleFormatError,
    data_cksum,
    parse_rebus_table,
    serialize_rebus_table,
)
from blacksquare.types import SpecialCellValue


class TestPuzChecksums:
    def test_data_cksum_calculation(self):
        # Known test data
        assert data_cksum(b"") == 0
        assert data_cksum(b"A") == 0x0041
        # Two bytes: 0x41 rotated right becomes 0x8020 + 0x42 = 0x8062
        assert data_cksum(b"AB") == 0x8062

    def test_rebus_table_serialization(self):
        d = {0: "HEART", 1: "SUN/MOON", 2: "STAR"}
        serialized = serialize_rebus_table(d)
        assert " 0:HEART;" in serialized
        assert " 1:SUN/MOON;" in serialized
        assert " 2:STAR;" in serialized

        parsed = parse_rebus_table(serialized)
        assert parsed == d


class TestPuzRoundtrip:
    def test_basic_puz_roundtrip(self):
        xw = Crossword(3, 4, symmetry=None)
        xw[0, 0] = BLACK
        xw[ACROSS, 1] = "BCD"
        xw[ACROSS, 4] = "ABCD"
        xw[ACROSS, 5] = "EFGH"
        xw[ACROSS, 1].clue = "Alphabet part 1"
        xw[ACROSS, 4].clue = "Alphabet part 2"
        xw[ACROSS, 5].clue = "Alphabet part 3"
        xw[DOWN, 1].clue = "First down"
        xw[DOWN, 2].clue = "Second down"
        xw[DOWN, 3].clue = "Third down"

        raw_bytes = xw.to_puz(
            title="Alphabet Puzzle",
            author="Tester",
            copyright="2026 Test Inc",
            notes="Enjoy the puzzle!",
        )
        assert isinstance(raw_bytes, bytes)
        assert raw_bytes.startswith(b"\0\0ACROSS&DOWN\0") or b"ACROSS&DOWN" in raw_bytes

        # Re-import from bytes
        loaded = Crossword.from_puz(raw_bytes)
        assert loaded.num_rows == 3
        assert loaded.num_cols == 4
        assert loaded[0, 0] == BLACK
        assert loaded[ACROSS, 1].value == "BCD"
        assert loaded[ACROSS, 4].value == "ABCD"
        assert loaded[ACROSS, 5].value == "EFGH"
        assert loaded[ACROSS, 1].clue == "Alphabet part 1"
        assert loaded[ACROSS, 4].clue == "Alphabet part 2"
        assert loaded[ACROSS, 5].clue == "Alphabet part 3"
        assert loaded[DOWN, 1].clue == "First down"
        assert loaded[DOWN, 2].clue == "Second down"
        assert loaded[DOWN, 3].clue == "Third down"

    def test_file_and_stream_io(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "HELLO"
        xw[ACROSS, 1].clue = "Greeting"

        # 1. Test with BytesIO stream
        buf = io.BytesIO()
        xw.to_puz(buf)
        buf.seek(0)
        loaded_stream = Crossword.from_puz(buf)
        assert loaded_stream[ACROSS, 1].value == "HELLO"
        assert loaded_stream[ACROSS, 1].clue == "Greeting"

        # 2. Test with file path
        with tempfile.NamedTemporaryFile(suffix=".puz", delete=False) as f:
            filepath = f.name

        try:
            xw.to_puz(filepath)
            assert os.path.exists(filepath)
            loaded_file = Crossword.from_puz(filepath)
            assert loaded_file[ACROSS, 1].value == "HELLO"
            assert loaded_file[ACROSS, 1].clue == "Greeting"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestPuzEdgeCases:
    def test_rebus_cells_roundtrip(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("HEART")
        xw[2, 0] = Rebus(across="SUN", down="MOON")
        xw[ACROSS, 1] = "AB(HEART)CD"
        xw[DOWN, 3] = "(HEART)XMZW"
        xw[ACROSS, 7] = "(SUN)LMNO"
        xw[DOWN, 1] = "AF(MOON)PQ"

        raw_bytes = xw.to_puz()
        puz_data = PuzData.from_bytes(raw_bytes)

        # Check that GRBS and RTBL extensions exist
        assert ExtensionCode.Rebus in puz_data.extensions
        assert ExtensionCode.RebusSolutions in puz_data.extensions

        # Re-import and verify rebuses are restored
        loaded = Crossword.from_puz(raw_bytes)
        assert isinstance(loaded[0, 2].value, Rebus)
        assert loaded[0, 2].value == Rebus("HEART")
        assert loaded[ACROSS, 1].value == "AB(HEART)CD"
        assert loaded[DOWN, 3].value == "(HEART)XMZW"

        assert isinstance(loaded[2, 0].value, Rebus)
        assert loaded[2, 0].value.across == "SUN"
        assert loaded[2, 0].value.down == "MOON"
        assert loaded[ACROSS, 7].value == "(SUN)LMNO"
        assert loaded[DOWN, 1].value == "AF(MOON)PQ"

    def test_markup_circled_cells_roundtrip(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "MAGIC"
        xw[1, 1].circled = True
        xw[3, 3].circled = True

        raw_bytes = xw.to_puz()
        puz_data = PuzData.from_bytes(raw_bytes)
        assert ExtensionCode.Markup in puz_data.extensions
        gext = puz_data.extensions[ExtensionCode.Markup]
        assert gext[1 * 5 + 1] & GridMarkup.Circled
        assert gext[3 * 5 + 3] & GridMarkup.Circled
        assert not (gext[0] & GridMarkup.Circled)

        # Re-import
        loaded = Crossword.from_puz(raw_bytes)
        assert loaded[1, 1].circled is True
        assert loaded[3, 3].circled is True
        assert loaded[0, 0].circled is False

    def test_empty_and_partial_fill_roundtrip(self):
        xw = Crossword(4, 4, symmetry=None)
        xw[0, 0] = "A"
        xw[0, 1] = EMPTY
        xw[0, 2] = "C"
        xw[0, 3] = BLACK

        raw_bytes = xw.to_puz()
        loaded = Crossword.from_puz(raw_bytes)
        assert loaded[0, 0].value == "A"
        assert loaded[0, 1].value == SpecialCellValue.EMPTY
        assert loaded[0, 2].value == "C"
        assert loaded[0, 3] == BLACK

    def test_invalid_puz_raises_error(self):
        with pytest.raises(PuzzleFormatError):
            PuzData.from_bytes(b"NOT_A_VALID_PUZ_HEADER_DATA")

        with pytest.raises(PuzzleFormatError):
            Crossword.from_puz(b"CORRUPTED_BYTES")

    def test_strict_checksum_verification(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "HELLO"
        raw_bytes = bytearray(xw.to_puz(title="Test", author="Tester"))

        # Valid bytes pass strict check
        puz_valid = PuzData.from_bytes(bytes(raw_bytes), strict=True)
        assert puz_valid.title == "Test"

        # Corrupt one byte in the solution grid (offset 52)
        corrupted = bytearray(raw_bytes)
        corrupted[52] = ord("X") if corrupted[52] != ord("X") else ord("Y")
        with pytest.raises(PuzzleFormatError, match="checksum"):
            PuzData.from_bytes(bytes(corrupted), strict=True)

    def test_15x15_rebus_and_markup_roundtrip(self):
        xw15 = Crossword(15, 15)
        blacks = [
            (0, 4),
            (1, 4),
            (2, 4),
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 7),
            (4, 7),
            (5, 7),
            (9, 7),
            (10, 7),
            (11, 7),
            (12, 4),
            (13, 4),
            (14, 4),
            (12, 10),
            (13, 10),
            (14, 10),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 12),
            (7, 13),
            (7, 14),
            (4, 0),
            (5, 0),
            (6, 0),
            (8, 14),
            (9, 14),
            (10, 14),
        ]
        for r, c in blacks:
            xw15[r, c] = BLACK

        for r in range(15):
            for c in range(15):
                if xw15[r, c] != BLACK:
                    xw15[r, c] = chr(65 + (r * 5 + c * 3) % 26)

        xw15[0, 2] = Rebus("HEART")
        xw15[2, 7] = Rebus(across="SUN", down="MOON")
        xw15[7, 7] = Rebus("STAR")
        xw15[14, 12] = Rebus("HEART")

        xw15[1, 1].circled = True
        xw15[13, 13].circled = True

        for i, w in enumerate(xw15.iterwords()):
            w.clue = f"Clue #{i + 1} for {w.number} {w.direction.value}"

        raw = xw15.to_puz(
            title="15x15 Thematic Crossword",
            author="Author Name",
            copyright="2026",
            notes="Enjoy the theme!",
        )

        puz_obj = PuzData.from_bytes(raw, strict=True)
        assert puz_obj.title == "15x15 Thematic Crossword"
        assert puz_obj.author == "Author Name"

        loaded = Crossword.from_puz(raw)
        assert loaded.num_rows == 15
        assert loaded.num_cols == 15
        assert loaded[0, 2].value == Rebus("HEART")
        assert loaded[2, 7].value == Rebus(across="SUN", down="MOON")
        assert loaded[7, 7].value == Rebus("STAR")
        assert loaded[14, 12].value == Rebus("HEART")
        assert loaded[1, 1].circled is True
        assert loaded[13, 13].circled is True
        assert loaded[0, 0].circled is False

        # Verify clues match
        orig_clues = {w.index: w.clue for w in xw15.iterwords()}
        loaded_clues = {w.index: w.clue for w in loaded.iterwords()}
        assert orig_clues == loaded_clues
