import pytest

from blacksquare import (
    ACROSS,
    BLACK,
    Crossword,
    CrosswordStats,
    Rebus,
    Symmetry,
)


class TestCrosswordCheck:
    def test_valid_grid_passes(self):
        xw5 = Crossword(5, 5, symmetry=Symmetry.ROTATIONAL)
        xw5[ACROSS, 1] = "SCARS"
        xw5[ACROSS, 6] = "TORES"
        xw5[ACROSS, 7] = "AREAS"
        xw5[ACROSS, 8] = "RESTS"
        xw5[ACROSS, 9] = "TESTS"
        res = xw5.check()
        assert res.is_valid is True
        assert bool(res) is True
        assert xw5.is_valid() is True
        assert len(res.errors) == 0

    def test_short_word_segment_detected(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 1] = BLACK  # Leaves (0, 0) as a 1-letter piece
        res = xw.check()
        assert res.is_valid is False
        assert bool(res) is False
        assert any("length 1 is shorter than minimum 3" in e for e in res.errors)

        # 2-letter segment
        xw2 = Crossword(5, 5, symmetry=None)
        xw2[0, 2] = BLACK  # Leaves (0, 0)..(0, 1) as a 2-letter piece
        res2 = xw2.check()
        assert res2.is_valid is False
        assert any("length 2 is shorter than minimum 3" in e for e in res2.errors)

    def test_custom_min_word_length(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 3] = BLACK
        xw[1, 3] = BLACK
        xw[2, 3] = BLACK
        xw[3, 3] = BLACK
        xw[4, 3] = BLACK
        # Cols 0..2 have 3-letter across words
        res_default = xw.check(min_word_length=3, require_connected=False)
        # Cols 4 is length 1 piece, so it fails default 3
        assert any("length 1" in e for e in res_default.errors)

    def test_symmetry_violation_detected(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 0] = BLACK
        # Symmetrical partner (4, 4) is not black

        res = xw.check(symmetry=Symmetry.ROTATIONAL)
        assert res.is_valid is False
        assert any("Symmetry violation" in e for e in res.errors)
        assert any("(0, 0)" in e or "(4, 4)" in e for e in res.errors)

    def test_duplicate_word_reuse_detected(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[ACROSS, 1] = "ALPHA"
        xw[ACROSS, 6] = "ALPHA"  # Reused word

        res = xw.check(allow_duplicates=False)
        assert res.is_valid is False
        assert any("Duplicate word 'ALPHA' reused" in e for e in res.errors)

        # allow_duplicates=True ignores duplicates
        res_allowed = xw.check(allow_duplicates=True)
        assert res_allowed.is_valid is True

    def test_duplicate_rebus_word_detected(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 2] = Rebus("HEART")
        xw[2, 2] = Rebus("HEART")
        xw[ACROSS, 1] = "AB(HEART)CD"
        xw[ACROSS, 7] = "AB(HEART)CD"  # Reused rebus word

        res = xw.check(allow_duplicates=False)
        assert res.is_valid is False
        assert any("Duplicate word 'AB(HEART)CD' reused" in e for e in res.errors)

    def test_disconnected_grid_detected(self):
        xw = Crossword(5, 5, symmetry=None)
        # Block off cell (0, 0)
        xw[0, 1] = BLACK
        xw[1, 0] = BLACK

        res = xw.check(require_connected=True)
        assert res.is_valid is False
        assert any("disconnected" in e for e in res.errors)

    def test_raise_on_error(self):
        xw = Crossword(5, 5, symmetry=None)
        xw[0, 1] = BLACK
        with pytest.raises(ValueError, match="Validation failed"):
            xw.check(raise_on_error=True)

    def test_require_filled(self):
        xw = Crossword(5, 5, symmetry=None)
        # Grid is empty/unfilled
        res = xw.check(require_filled=True)
        assert res.is_valid is False
        assert any("empty / open cell" in e for e in res.errors)


class TestCrosswordStats:
    def test_stats_counts(self):
        xw = Crossword(5, 5, symmetry=Symmetry.ROTATIONAL)
        xw[2, 2] = BLACK
        xw[0, 0] = "A"
        xw[0, 1] = "B"
        xw[0, 2] = "C"
        xw[0, 3] = "D"
        xw[0, 4] = "E"
        xw[1, 1].circled = True
        xw[3, 3].shaded = True
        xw[4, 4] = Rebus("STAR")

        stats = xw.stats()
        assert isinstance(stats, CrosswordStats)
        assert stats.total_cells == 25
        assert stats.black_squares == 1
        assert stats.open_cells == 24
        assert stats.black_square_pct == 4.0
        assert stats.open_cell_pct == 96.0

        assert stats.rebus_count == 1
        assert stats.circled_count == 1
        assert stats.shaded_count == 1

        assert stats.total_words == 12
        assert stats.across_words == 6
        assert stats.down_words == 6

        # Letter counts
        assert stats.letter_counts["A"] == 1
        assert stats.letter_counts["B"] == 1
        assert stats.letter_counts["STAR"] == 1

        # Dict access and conversion
        d = stats.to_dict()
        assert d["total_words"] == 12
        assert d["black_squares"] == 1
        assert stats["total_words"] == 12

        # String representation
        stats_str = str(stats)
        assert "CrosswordStats:" in stats_str
        assert "Total Words: 12" in stats_str
        assert "Black Squares: 1 / 25" in stats_str
        assert "STAR: 1" in stats_str
