import pytest

from blacksquare import ACROSS, DOWN, Crossword, WordList
from blacksquare.word_list import MatchWordList


class TestWordList:
    def test_from_list(self):
        words = ["ABC", "AAA", "ABCD", "ABCDE"]
        word_list = WordList(words)
        assert word_list.words == sorted(words)
        assert all([s == 1.0 for s in word_list.scores])

    def test_from_dict(self):
        word_dict = {"aaa": 10, "bBB": 100, "C CC": 50}
        word_list = WordList(word_dict)
        assert word_list.words == ["BBB", "CCC", "AAA"]
        assert word_list.scores == [1.0, 0.5, 0.1]

    def test_from_file(self, tmp_path):
        dict_file = tmp_path / "wordlist.dict"
        with dict_file.open("w") as f:
            f.writelines(["aaa;10\n", "bBB;100\n", "C CC;50\n"])
        word_list = WordList(dict_file)
        assert word_list.words == ["BBB", "CCC", "AAA"]
        assert word_list.scores == [1.0, 0.5, 0.1]

    @pytest.mark.parametrize("blank_string", ["?", " ", "_"])
    def test_find_matches_str(self, word_list, blank_string):
        matches = word_list.find_matches_str(f"{blank_string}BB")
        assert matches.words == ["ABB", "BBB"]

    @pytest.mark.parametrize("no_match_str", ["ZZZ", "Z??", "ABCDABCD?"])
    def test_no_matches(self, word_list, no_match_str):
        assert len(word_list.find_matches_str(no_match_str)) == 0

    def test_find_matches(self, xw, word_list):
        across_matches = word_list.find_matches(xw[ACROSS, 5])
        assert across_matches.words == ["ABB", "ABC", "CCB", "BBB", "BCD"]

        down_matches = word_list.find_matches(xw[DOWN, 1])
        assert down_matches.words == ["BBB"]

    def test_find_matches_filled(self, xw: Crossword, word_list: WordList):
        matches = word_list.find_matches(xw[ACROSS, 1])
        assert matches.words == ["BCD"]

    def test_get_score(self, word_list):
        assert word_list.get_score("BBB") == 0.1
        assert word_list.get_score("XYZ") is None

    def test_add_word_list(self):
        word_list_1 = {"ZZZ": 0.1, "AAA": 1.0}
        word_list_2 = {"ABC": 0.9, "XYZ": 1.0}
        new_word_list = WordList(word_list_1) + WordList(word_list_2)
        assert new_word_list.words == ["AAA", "XYZ", "ABC", "ZZZ"]
        assert new_word_list.scores == [1.0, 1.0, 0.9, 0.1]


class TestMatchWordList:
    @pytest.fixture
    def matches(self, word_list) -> MatchWordList:
        return word_list.find_matches_str("???")

    def test_letter_scores_at_index(self, matches: MatchWordList):
        assert matches.letter_scores_at_index(1) == {"B": 2.1, "C": 1.1}

    def test_rescore(self, matches: MatchWordList):
        def rescore_fn(word, score):
            if "A" in word:
                return 2.0 * score
            else:
                return 1.0 * score

        rescored = matches.rescore(rescore_fn)
        assert rescored.get_score("ABC") == 2.0
        assert rescored.get_score("BCD") == 0.1
