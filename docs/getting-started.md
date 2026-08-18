# Getting Started

## Installation

You can install `blacksquare` using `pip` or `uv`:

=== "uv (Recommended)"

    ```bash
    uv add blacksquare
    ```

=== "pip"

    ```bash
    pip install blacksquare
    ```

=== "With PDF Export Support"

    ```bash
    pip install "blacksquare[pdf]"
    ```

---

## Creating Your First Puzzle

### 1. Initialize a Grid

You can create an empty square or rectangular grid by specifying dimensions:

```python
from blacksquare import Crossword, Symmetry

# 15x15 standard weekday crossword with rotational symmetry
xw = Crossword(15, 15, symmetry=Symmetry.ROTATIONAL)
```

### 2. Add Black Squares

Black squares can be assigned using `(row, col)` coordinates. Because symmetry is enabled by default, setting a black block automatically updates the symmetric cell:

```python
from blacksquare import BLACK

xw[0, 4] = BLACK  # Also places BLACK at [14, 10] symmetrically
```

### 3. Inspect the Grid

Print your grid to the console with numbers and clues:

```python
xw.pprint(numbers=True)
```

In Jupyter notebooks, simply typing `xw` on the last line renders a styled HTML crossword grid.

---

## Next Steps

- Explore [Grids & Cells](user-guide/grid-and-cells.md) for coordinate systems and cell styling.
- Explore [Auto-Fill](user-guide/filling.md) for constraint-based grid filling.
- Explore [Export & Publishing](user-guide/import-export.md) to generate `.puz` and `.pdf` files.
