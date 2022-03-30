# Blacksquare
![build and test](https://github.com/pmaher86/blacksquare/actions/workflows/build-and-test.yaml/badge.svg)

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It also has tools for finding valid fills, and HTML rendering that plugs nicely into Jupyter notebooks. Blacksquare supports import and export from the .puz format via [puzpy](https://github.com/alexdej/puzpy), as well as .pdf export in the [New York Times submission format](https://www.nytimes.com/puzzles/submissions/crossword) (requires [wkhtmltopdf](https://wkhtmltopdf.org/)).

## Native HTML rendering in Jupyter
![Jupyter example](assets/jupyter.png?raw=true)

## Basic features
The interface is built to use Python's indexing syntax to express high-level crossword concepts.

```python
>>> from blacksquare import Crossword, BLACK, EMPTY, ACROSS, DOWN
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
>>> xw[DOWN, 3] = xw[DOWN, 3].find_matches()[0]
>>> xw.pprint()
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │ A │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ A │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ C │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│ D │ O │ E │███│   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ L │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ L │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ S │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
# We can also index into Word objects
>>> xw[DOWN, 3][1] = 'D'
>>> xw[DOWN, 3][0] = EMPTY
>>> xw[DOWN, 3].value
' DCELLS'
```
Puzzles can be imported and exported easily.
```python
>>> xw.to_puz('puzzle.puz')
>>> xw = Crossword.from_puz('puzzle.puz')
>>> xw.to_pdf('puzzle.pdf', header=['Name', 'Address', 'Email'])
```
There are useful utility functions for navigating.
```python
>>> unfilled_words = [w for w in xw.iterwords() if w.is_open()]
>>> xw[DOWN,13]crosses
[Across 12: "??L????", Across 14: "??L????", Across 15: "??S????"]
```
Clues can be attached to words.
```python
>>> xw[ACROSS, 10].clue = "A deer, a female deer"
>>> xw.clues
{(<Across>, 1): '', (<Across>, 8): '', (<Across>, 9): '', (<Across>, 10): 'A deer, a female deer', (<Across>, 11): '', (<Across>, 12): '', (<Down>, 1): '', (<Down>, 2): '', (<Down>, 3): '', (<Down>, 4): '', (<Down>, 5): '', (<Down>, 6): '', (<Down>, 7): '', (<Across>, 14): '', (<Across>, 15): '', (<Down>, 13): ''}
```
You can also make modifications that return new grid objects, to support things like custom branching searches.
```python
>>> new_xw = xw.set_word((ACROSS, 11), 'ABC', inplace=False)
```

A main attraction are the utilities to help find valid fills. It implements an algorithm called `cross_rank` that tries to maximize the number of valid matches for each cross. Multiple clues can searched at once for mutually compatible solutions using a beam search method, and entire continuous unfilled areas can be searched from a single seed clue.
```python
>>> matches = xw[DOWN, 1].find_matches()
>>> matches[0]
'MCADAMS'
# This returns a list of new Crossword objects sorted by the dictionary scores of the words
>>> matching_grids = xw.find_solutions([(DOWN, 1), (DOWN, 2)])
# This returns a list of Crossword objects with everything contiguous to (DOWN, 1) filled.
# Careful, this can be slow for large unfilled areas!
>>> matching_grids = xw.find_area_solutions((DOWN, 1))
```


Custom word lists are supported and can be passed into the `Crossword` constructor or any of the solving methods. The default word list used is the [Crossword Nexus Collaborative Word List](https://github.com/Crossword-Nexus/collaborative-word-list).
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
- [ ] Annotations for themed puzzles (circles etc.)
