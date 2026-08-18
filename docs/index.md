# Blacksquare

<p align="center">
  <strong>An intuitive, high-performance Python package for crossword creation, manipulation, and solving backed by an ultra-fast Rust engine.</strong>
</p>

---

## Quick Example

```python
from blacksquare import Crossword, ACROSS, DOWN, BLACK, EMPTY

# 1. Initialize a 7x7 crossword with rotational symmetry
xw = Crossword(num_rows=7)

# 2. Place black squares (automatically mirrored symmetrically)
xw[3, 3] = BLACK

# 3. Fill specific theme entries
xw[ACROSS, 10] = "DOE"

# 4. Search for valid crosses and auto-fill remaining grid in milliseconds
solved_xw = xw.fill()

# 5. Print to terminal or export to NYT submission PDF
solved_xw.pprint()
solved_xw.to_pdf("my_crossword.pdf", header=["Author Name", "contact@example.com"])
```
