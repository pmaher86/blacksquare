![build and test](https://github.com/pmaher86/blacksquare/actions/workflows/build-and-test.yaml/badge.svg)

Blacksquare is a Python package for crossword creators. It aims to be an intuitive interface for working with crossword puzzles programmatically. It also has tools for finding valid fills, and HTML rendering that plugs nicely into Jupyter notebooks. Blacksquare supports import and export from the .puz format via [puzpy](https://github.com/alexdej/puzpy), as well as .pdf export in the New York Times submission format (requires [wkhtmltopdf](https://wkhtmltopdf.org/)).

The default dictionary used is [Peter Broda's scored  wordlist](https://peterbroda.me/crosswords/wordlist/).