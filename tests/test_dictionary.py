from blacksquare.crossword import Crossword
from blacksquare.dictionary import parse_dictionary


class TestDictionary:
    def test_default_dictionary(self):
        xw = Crossword(3)
        assert "ALFALFA" in xw._dictionary
        assert all([0 <= v <= 1 for v in xw._dictionary.values()])

    def test_parse_dictionary(self):
        dict_1 = ["Aa A", "ABC", "XYZ"]
        assert parse_dictionary(dict_1)["AAA"] == 1.0
        dict_2 = {"Aa A": 5, "ABC": 0, "XYZ": 10}
        assert parse_dictionary(dict_2)["AAA"] == 0.5
        # idempotence
        assert parse_dictionary(parse_dictionary(dict_2)) == parse_dictionary(dict_2)
