# Words & Crosses

## Word Indexing

Words in `blacksquare` are addressed using `(Direction, number)` tuples:

```python
from blacksquare import Crossword, ACROSS, DOWN

xw = Crossword(15)

# Set 1-Across to 'HELLO'
xw[ACROSS, 1] = "HELLO"

# Access the Word object
w = xw[ACROSS, 1]
print(w.value)   # 'HELLO'
print(len(w))    # 5
```

## Traversing Crosses

Every word slot knows which crossing words intersect each of its letters:

```python
w = xw[ACROSS, 1]
for i, cross in enumerate(w.crosses):
    if cross is not None:
        print(f"Letter {i} ('{w.value[i]}') crosses {cross.direction.value} {cross.number}")
```

## Assigning Clues

Clues can be attached directly to word objects or viewed as a dictionary:

```python
xw[ACROSS, 1].clue = "Greeting"
print(xw.clues)
```

## Iterating Over Words and Cells

```python
# Iterate over all open (unfilled) across words
for word in xw.iterwords(direction=ACROSS, only_open=True):
    print(word.index, word.value)

# Iterate over all cells in row-major order
for cell in xw.itercells():
    if cell.is_black():
        pass
```
