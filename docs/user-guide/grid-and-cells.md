# Grids & Cells

## Coordinate Indexing

Individual cells in a crossword are accessed via 0-indexed `(row, col)` tuples:

```python
from blacksquare import Crossword, BLACK, EMPTY

xw = Crossword(15)

# Place a black square at row 0, column 4
xw[0, 4] = BLACK

# Read the Cell object at (0, 0)
cell = xw[0, 0]
print(cell.value)  # SpecialCellValue.EMPTY
```

## Cell Values and Properties

A cell can hold an uppercase letter or a special value (`BLACK`, `EMPTY`):

```python
xw[0, 0] = "C"
xw[0, 1] = "A"
xw[0, 2] = "T"
```

### Visual Highlights (Shading & Circles)

You can shade or circle individual cells for themed puzzles:

```python
# Circle the cell at (1, 1)
xw[1, 1].circled = True

# Shade the cell at (2, 2)
xw[2, 2].shaded = True
```

## Symmetry Modes

`blacksquare` supports multiple grid symmetry options from the `Symmetry` enum:

- `Symmetry.ROTATIONAL` (180° rotational symmetry — standard American crossword)
- `Symmetry.FULL` (4-way rotational and reflective symmetry)
- `Symmetry.VERTICAL` (Left-right mirror reflection)
- `Symmetry.HORIZONTAL` (Top-bottom mirror reflection)
- `Symmetry.BIAXIAL` (Both horizontal and vertical reflection)
- `Symmetry.NE_DIAGONAL` / `Symmetry.NW_DIAGONAL` (Diagonal reflection)

```python
xw = Crossword(15, symmetry=Symmetry.VERTICAL)
```
