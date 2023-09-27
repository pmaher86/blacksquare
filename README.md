# Blacksquare
![Build Status](https://github.com/pmaher86/blacksquare/actions/workflows/build-and-test.yaml/badge.svg) ![Documentation Status](https://readthedocs.org/projects/blacksquare/badge/?version=latest)

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It also has tools for finding valid fills, and HTML rendering that plugs nicely into Jupyter notebooks. Blacksquare supports import and export from the .puz format via [puzpy](https://github.com/alexdej/puzpy), as well as .pdf export in the [New York Times submission format](https://www.nytimes.com/puzzles/submissions/crossword) (requires [wkhtmltopdf](https://wkhtmltopdf.org/)).

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


Custom word lists are supported and can be passed into the `Crossword` constructor or any of the solving methods. The default word list used is the [Crossword Nexus Collaborative Word List](https://github.com/Crossword-Nexus/collaborative-word-list).

## Example: full symmetry puzzles
As an example of how blacksquare's abstractions allow for non-trivial crossword construction, consider the [June 6 2023 NYT puzzle](https://www.xwordinfo.com/Crossword?date=6/6/2023), which displays not only a rotationaly symmetric grid but a rotationally symmetric *fill*. While this might seem daunting to build, all we have to do is override the `set_word` method of `Crossword` to fill two words at once, and then restrict our wordlist to emordnilaps (words that are also a word when reversed). 
```python
class SymmetricCrossword(Crossword):
    def set_word(self, word_index: WordIndex, value: str) -> None:
        super().set_word(word_index, value)
        super().set_word(self.get_symmetric_word_index(word_index), value[::-1])

emordilaps = {}
for word, score in tqdm(bs.DEFAULT_WORDLIST):
    reverse_score = bs.DEFAULT_WORDLIST.get_score(word[::-1])
    if reverse_score:
        emordilaps[word] = min(score, reverse_score)
emordilaps_wordlist = bs.WordList(emordilaps)

# Now just construct the puzzle and fill!
xw = SymmetricCrossword(15)
filled = [
    (0, 3), (0, 4), (0, 5), (0, 11), (1, 4), (1, 5), (1, 11),
    (2, 4), (2, 11), (3, 4), (3, 9), (4, 0), (4, 1), (4, 2),
    (4, 7), (4, 8), (4, 14), (5, 6), (5, 12), (5, 13), (5, 14), 
    (6, 5), (6, 10), (7, 3),
]
for i in filled:
    xw[i] = bs.BLACK
xw.fill(emordnilap_wordlist, temperature=0.3)

┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│^F │^E │^N │███│███│███│^S │^N │^I │^P │^S │███│^E │^D │^A │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^L │ I │ A │^R │███│███│^P │ O │ S │ E │ A │███│^V │ E │ R │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^O │ K │ I │ E │███│^R │ E │ W │ A │ R │ D │███│^A │ L │ B │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^G │ O │ R │ T │███│^A │ T │ I │ N │███│^D │^E │ L │ I │ A │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│███│███│^R │^A │ P │ S │███│███│^R │ E │ E │ S │ A │███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^S │^T │^R │ O │ P │ S │███│^S │^P │ A │ N │ K │███│███│███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^R │ E │ E │ S │ A │███│^S │ T │ O │ M │███│^S │^E │^P │^S │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^A │ R │ M │███│^R │^O │ T │ A │ T │ O │^R │███│^M │ R │ A │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^S │ P │ E │^S │███│^M │ O │ T │ S │███│^A │^S │ E │ E │ R │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│███│███│^K │^N │ A │ P │ S │███│^S │ P │ O │ R │ T │ S │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│███│^A │^S │ E │ E │ R │███│███│^S │ P │ A │ R │███│███│███│
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^A │ I │ L │ E │ D │███│^N │^I │ T │ A │███│^T │^R │^O │^G │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^B │ L │ A │███│^D │^R │ A │ W │ E │ R │███│^E │ I │ K │ O │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^R │ E │ V │███│^A │ E │ S │ O │ P │███│███│^R │ A │ I │ L │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│^A │ D │ E │███│^S │ P │ I │ N │ S │███│███│███│^N │ E │ F │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```
There's clearly some extra curation that could be done to improve the word list, and we'd need a little more logic to avoid repeat fills and using true palindromes outside of the center. But not bad for a few lines of code!

## Installation
`pip install blacksquare`

You'll also need to install [wkhtmltopdf](https://wkhtmltopdf.org/) for .pdf export to work.

## Future plans
Blacksquare is not a GUI application and isn't intended to be one. Blacksquare is also not a package for solving crossword puzzles. 

Possible extensions include: 
- [ ] Other file formats
- [ ] Other/better filling heuristics
- [ ] Verifying puzzle validity
- [ ] Rebuses
- [x] Annotations for themed puzzles (circles etc.)
