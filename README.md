# Blacksquare
![Build Status](https://github.com/pmaher86/blacksquare/actions/workflows/build-and-test.yaml/badge.svg) ![Documentation Status](https://readthedocs.org/projects/blacksquare/badge/?version=latest)

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It also has tools for finding valid fills, and HTML rendering that plugs nicely into Jupyter notebooks. Blacksquare supports import and export from the .puz format via [puzpy](https://github.com/alexdej/puzpy), as well as .pdf export in the [New York Times submission format](https://www.nytimes.com/puzzles/submissions/crossword) (requires the [pdf] extra).

## Native HTML rendering in Jupyter
![Jupyter example](assets/jupyter.png?raw=true)

## Basic features
The interface is built to use Python's indexing syntax to express high-level crossword concepts.

```python
>>> from blacksquare import Crossword, BLACK, EMPTY, ACROSS, DOWN, DEFAULT_WORDLIST
>>> xw = Crossword(num_rows=7)
# (row, column) indexing for individual cells
>>> xw[3,3] = BLACK
>>> xw.pprint(numbers=True)
┌───┬───┬───┬───┬───┬───┬───┐
│1  │2  │3  │4  │5  │6  │7  │
├───┼───┼───┼───┼───┼───┼───┤
│8  │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│9  │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│10 │   │   │███│11 │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│12 │   │   │13 │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│14 │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│15 │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
# (direction, number) indexing for words
>>> xw[ACROSS, 10] = 'DOE'
>>> xw[DOWN, 3] = xw[DOWN, 3].find_matches().words[0]
>>> xw.pprint()
┌───┬───┬───┬───┬───┬───┬───┐
│^  │^  │^C │^  │^  │^  │^  │
├───┼───┼───┼───┼───┼───┼───┤
│^  │   │ A │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│^  │   │ V │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│^D │ O │ E │███│^  │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│^  │   │ A │^  │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│^  │   │ R │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│^  │   │ T │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
# We can also index into Word objects
>>> xw[DOWN, 3][1] = 'I'
>>> xw[DOWN, 3][0] = EMPTY
>>> xw[DOWN, 3].value
' IVEART'
```
Puzzles can be imported and exported easily.
```python
>>> xw.to_puz('puzzle.puz')
>>> xw = Crossword.from_puz('puzzle.puz')
>>> xw.to_pdf('puzzle.pdf', header=['Name', 'Address', 'Email'])
```
There are useful utility functions for navigating.
```python
>>> unfilled_words = list(xw.iterwords(only_open=True))
>>> xw[DOWN, 13].crosses
[Word(Across 12: "??A????)",
 Word(Across 14: "??R????)",
 Word(Across 15: "??T????)"]

```
Clues can be attached to words.
```python
>>> xw[ACROSS, 10].clue = "A deer, a female deer"
>>> xw.clues
{(<Across>, 1): '',
 (<Across>, 8): '',
 (<Across>, 9): '',
 (<Across>, 10): 'A deer, a female deer',
 (<Across>, 11): '',
 (<Across>, 12): '',
 (<Across>, 14): '',
 (<Across>, 15): '',
 (<Down>, 1): '',
 (<Down>, 2): '',
 (<Down>, 3): '',
 (<Down>, 4): '',
 (<Down>, 5): '',
 (<Down>, 6): '',
 (<Down>, 7): '',
 (<Down>, 13): ''}
```
You can also copy grid objects, to support things like custom branching searches.
```python
>>> new_xw = xw.copy()
>>> new_xw[ACROSS, 11] = 'ABC'
```

A main attraction are the utilities to help find valid fills. It implements an algorithm called `cross_rank` that tries to maximize the number of valid matches for each cross. Multiple clues can searched at once for mutually compatible solutions using a beam search method, and entire continuous unfilled areas can be searched from a single seed clue.
```python
>>> matches = xw[DOWN, 1].find_matches()
>>> matches[0]
ScoredWord(word='MACDUFF', score=25.387070408819042)
# This returns a new valid Crossword fill, with optional randomness and word list control.
>>> filled = xw.fill(temperature=1, word_list=DEFAULT_WORDLIST.score_filter(0.5))
```


Custom word lists are supported and can be passed into the `Crossword` constructor or any of the solving methods. The default word list used is from [spread the word(list)](https://www.spreadthewordlist.com/). (Please note that the word list carries a [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.)

## Example: full symmetry puzzles
As an example of how blacksquare's abstractions allow for non-trivial crossword construction, consider the [June 6 2023 NYT puzzle](https://www.xwordinfo.com/Crossword?date=6/6/2023), which displays not only a rotationaly symmetric grid but a rotationally symmetric *fill*. While this might seem daunting to build, all we have to do is override a couple methods of the base Crossword class, and use some modified wordlists.
```python
class SymmetricCrossword(bs.Crossword):
    # This sets symmetric words to be mirror images
    def set_word(self, word_index, value):
        super().set_word(word_index, value)
        super().set_word(self.get_symmetric_word_index(word_index), value[::-1])

    # This makes it so that we only track a unique half of the puzzle in the dependency
    #   graph (needed for the fill algorithm).
    def get_disconnected_open_subgrids(self):
        subgrids = super().get_disconnected_open_subgrids()
        new_subgrids = []
        for sg in subgrids:
            new_sg = sorted([min(i, self.get_symmetric_word_index(i)) for i in sg])
            if new_sg not in new_subgrids:
                new_subgrids.append(new_sg)
        return new_subgrids

palindromes = {}
emordnilaps = {}
for word, score in bs.DEFAULT_WORDLIST:
    if word == word[::-1]:
        palindromes[word] = score
    else:
        reverse_score = bs.DEFAULT_WORDLIST.get_score(word[::-1])
        if reverse_score:
            emordnilaps[word] = min(score, reverse_score)
palindrome_wordlist = bs.WordList(palindromes)
emordnilap_wordlist = bs.WordList(emordnilaps)

# Now construct the grid
xw = SymmetricCrossword(15)
filled = [
    (0, 3), (0, 4), (0, 5), (0, 11), (1, 4), (1, 5), (1, 11),
    (2, 4), (2, 11), (3, 4), (3, 9), (4, 0), (4, 1), (4, 2),
    (4, 7), (4, 8), (4, 14), (5, 6), (5, 12), (5, 13), (5, 14),
    (6, 5), (6, 10), (7, 3),
]
for i in filled:
    xw[i] = bs.BLACK

# Now fill the central words with palindromes
central_words = [w for w in xw.iterwords() if w.symmetric_image.index == w.index]
for cw in central_words:
    cw.value = palindrome_wordlist.find_matches(cw)[0].word

# And the rest!
xw.fill(emordnilap_wordlist, score_filter=0.3)

┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│^R │^O │^M │███│███│███│^A │^G │^A │^R │^D │███│^M │^E │^D │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^E │ D │ A │^M │███│███│^R │ A │ T │ E │ R │███│^A │ D │ U │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^N │ A │ N │ U │███│^L │ A │ M │ I │ N │ A │███│^O │ I │ C │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^E │ Y │ E │ R │███│^O │ M │ E │ N │███│^W │^A │ R │ T │ S │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│███│███│^D │^A │ T │ A │███│███│^S │ E │ R │ I │ A │███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^S │^L │^E │ E │ T │ S │███│^S │^M │ A │ R │ T │███│███│███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^T │ E │ R │ R │ A │███│^S │ T │ A │ B │███│^A │^S │^O │^P │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^E │ B │ O │███│^R │^A │ C │ E │ C │ A │^R │███│^O │ B │ E │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^P │ O │ S │^A │███│^B │ A │ T │ S │███│^A │^R │ R │ E │ T │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│███│███│^T │^R │ A │ M │ S │███│^S │ T │ E │ E │ L │ S │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│^A │^I │ R │ E │ S │███│███│^A │ T │ A │ D │███│███│███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^S │ T │ R │ A │ W │███│^N │^E │ M │ O │███│^R │^E │^Y │^E │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^C │ I │ O │███│^A │^N │ I │ M │ A │ L │███│^U │ N │ A │ N │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^U │ D │ A │███│^R │ E │ T │ A │ R │███│███│^M │ A │ D │ E │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^D │ E │ M │███│^D │ R │ A │ G │ A │███│███│███│^M │ O │ R │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```
There's clearly some extra curation that could be done to improve the word list, but not bad for a couple dozen lines of code!

## Installation
`pip install blacksquare`

or if you want to enable pdf export

`pip install blacksquare[pdf]`

## Future plans
Blacksquare is not a GUI application and isn't intended to be one. Blacksquare is also not a package for solving crossword puzzles.

Possible extensions include:
- [ ] Other file formats
- [ ] Other/better filling heuristics
- [ ] Verifying puzzle validity
- [ ] Rebuses
- [x] Annotations for themed puzzles (circles etc.)
