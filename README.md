# Blacksquare
![Build Status](https://github.com/pmaher86/blacksquare/actions/workflows/build-and-test.yaml/badge.svg) ![Documentation Status](https://readthedocs.org/projects/blacksquare/badge/?version=latest)

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It features high-performance grid solving powered by a Rust backend, rich HTML rendering that plugs nicely into Jupyter notebooks, native Across Lite (.puz) file import and export (with support for rebuses and circles), and .pdf export in the [New York Times submission format](https://www.nytimes.com/puzzles/submissions/crossword) (requires the [pdf] extra).

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
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┼───┼───┼───┤
│ 8 │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│ 9 │   │   │   │   │   │   │
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
# Only last digits of numbers fit in this view
>>> xw.pprint()
┌───┬───┬───┬───┬───┬───┬───┐
│¹  │²  │³B │⁴  │⁵  │⁶  │⁷  │
├───┼───┼───┼───┼───┼───┼───┤
│⁸  │   │ A │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│⁹  │   │ R │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│⁰D │ O │ E │███│¹  │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│²  │   │ X │³  │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│⁴  │   │ A │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│⁵  │   │ M │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
# We can also index into Word objects
>>> xw[DOWN, 3][0] = EMPTY
>>> xw[DOWN, 3].value
' AREXAM'
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
[Word(Across 12: "??X????"),
 Word(Across 14: "??A????"),
 Word(Across 15: "??M????")]

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

Rebuses, cell styling (circles and shading), and rule validation are supported out of the box.
```python
>>> from blacksquare import Rebus
>>> xw[0, 2] = Rebus("HEART")
>>> xw[1, 1].circled = True
>>> xw[2, 2].shaded = True
>>> xw.check()  # verifies connectivity, symmetry, word lengths, and duplicate fills
ValidationResult(is_valid=True, errors=[], warnings=[])
>>> xw.stats()  # returns grid statistics (word counts, open cells, letter frequencies, etc.)
```

A core feature of blacksquare are the utilities to help find valid fills, powered by a fast heuristic-guided backtracking solver written in Rust.
```python
>>> matches = xw[DOWN, 1].find_matches()
>>> matches[0]
ScoredWord(word='SANDBAG', score=26.863017541323376)
# This returns a new valid Crossword fill, with optional randomness and word list control.
>>> filled = xw.fill(temperature=1, word_list=DEFAULT_WORDLIST.score_filter(0.5))
```


Custom word lists are supported and can be passed into the `Crossword` constructor or any of the solving methods. The default word list used is from [spread the word(list)](https://www.spreadthewordlist.com/). (Please note that the word list carries a [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.)

## Installation
```bash
pip install blacksquare
```

or if you want to enable pdf export

```bash
pip install "blacksquare[pdf]"
```

## Development setup

Blacksquare requires Python 3.10+ and a [Rust toolchain](https://rustup.rs/).

1. Clone the repository:
   ```bash
   git clone https://github.com/pmaher86/blacksquare.git
   cd blacksquare
   ```

2. Install dependencies and compile the Rust extension in editable mode (using [`uv`](https://docs.astral.sh/uv/)):
   ```bash
   uv sync --all-groups --all-extras
   uv run maturin develop
   ```

3. Run the test suite:
   ```bash
   uv run pytest
   ```

4. Run code formatting, linting, and type checking:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check src
   ```

5. Build or serve documentation locally:
   ```bash
   uv run mkdocs serve
   ```
