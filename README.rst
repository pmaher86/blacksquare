###########
Blacksquare
###########

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It also has tools for finding valid fills, and HTML rendering that plugs nicely into Jupyter notebooks. Blacksquare supports import and export from the .puz format via `puzpy <https://github.com/alexdej/puzpy>`_, as well as .pdf export in the `New York Times submission format <https://www.nytimes.com/puzzles/submissions/crossword>`_ (requires `wkhtmltopdf <https://wkhtmltopdf.org/>`_).

**************
Basic features
**************
The interface is built to use Python's indexing syntax to express high-level crossword concepts.
::

    >>> from blacksquare import Crossword, black, empty, across, down
    >>> xw = Crossword(num_rows=7)
    # (row, column) indexing for individual cells
    >>> xw[3,3] = black
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
    >>> xw[across, 10] = 'DOE'
    >>> xw[down, 3] = xw[down, 3].find_matches()[0]
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
    >>> xw[down, 3][1] = 'D'
    >>> xw[down, 3][0] = empty
    >>> xw[down, 3].value
    ' DCELLS'

Puzzles can be imported and exported easily.
::

    >>> xw.to_puz('puzzle.puz')
    >>> xw = Crossword.from_puz('puzzle.puz')
    >>> xw.to_pdf('puzzle.pdf', header=['Name', 'Address', 'Email'])

There are useful utility functions for navigating.
::

    >>> unfilled_words = [w for w in xw.iterwords() if w.is_open()]
    >>> xw[down,13].get_crosses()
    [Across 12: "??L????", Across 14: "??L????", Across 15: "??S????"]

Clues can be attached to words.
::

    >>> xw[across, 10].clue = "A deer, a female deer"
    >>> xw.clues
    {(<Across>, 1): '', (<Across>, 8): '', (<Across>, 9): '', (<Across>, 10): 'A deer, a female deer', (<Across>, 11): '', (<Across>, 12): '', (<Down>, 1): '', (<Down>, 2): '', (<Down>, 3): '', (<Down>, 4): '', (<Down>, 5): '', (<Down>, 6): '', (<Down>, 7): '', (<Across>, 14): '', (<Across>, 15): '', (<Down>, 13): ''}

You can also make modifications that return new grid objects, to support things like custom branching searches.
::

    >>> new_xw = xw.set_word((across, 11), 'ABC', inplace=False)


A main attraction are the utilities to help find valid fills. It implements an algorithm called `cross_rank` that tries to maximize the number of valid matches for each cross. Multiple clues can searched at once for mutually compatible solutions using a beam search method, and entire continuous unfilled areas can be searched from a single seed clue.
::

    >>> matches = xw[down,1].find_matches(sort_method='cross_match')
    >>> matches[0]
    'MCADAMS'
    # This retuns a list of new Crossword objects sorted by the dictionary scores of the words
    >>> matching_grids = xw.find_solutions([(down, 1), (down, 2)])
    # This returns a list of Crossword objects with everything contiguous to (down, 1) filled.
    # Careful, this can be slow for large unfilled areas!
    >>> matching_grids = xw.find_area_solutions((down, 1))


Custom dictionaries are supported and can be passed into the `Crossword` constructor or any of the solving methods. The default dictionary used is `Peter Broda's scored wordlist <https://peterbroda.me/crosswords/wordlist/>`_.

************
Installation
************
::

    pip install blacksquare

You'll also need to install `wkhtmltopdf <https://wkhtmltopdf.org/>`_ for .pdf export to work.
