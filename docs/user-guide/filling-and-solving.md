# Solving & Auto-Fill

`blacksquare` includes an ultra-fast, Rust-powered backtracking solver that automatically fills empty regions of a crossword with valid words from the dictionary.

---

## Basic Fill

Calling `fill()` on a `Crossword` attempts to find a complete, valid solution:

```python
from blacksquare import Crossword, ACROSS, BLACK

xw = Crossword(7)
xw[3, 3] = BLACK
xw[ACROSS, 10] = "DOE"

# Solves the remainder of the grid in milliseconds
solved = xw.fill()

if solved is not None:
    solved.pprint()
else:
    print("No valid fill found.")
```

---

## Filling Parameters

`fill()` accepts several parameters to tune the search behavior:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `word_list` | `WordList` | `DEFAULT_WORDLIST` | The candidate word list to draw words from. |
| `timeout_secs` | `float` | `None` | Optional timeout in seconds. If exceeded, returns `None`. |
| `temperature` | `float` | `0.0` | Controls heuristic randomness ($0.0$ for deterministic highest-scoring fill, $> 0.0$ for diverse variations). |
| `score_filter` | `float` | `None` | Only consider dictionary words with a score $\ge$ threshold (e.g. $0.5$). |
| `allow_repeats` | `bool` | `False` | Whether duplicate words are allowed in the solution. |

### Example with Quality Filter and Temperature

```python
# Generate a diverse fill using only high-quality words (score >= 0.7)
solution = xw.fill(score_filter=0.7, temperature=0.5, timeout_secs=5.0)
```

---

## How It Works

1. **Minimum Remaining Values (MRV)**: Selects the most constrained slot first with fewest matching candidates.
2. **Fused Crossing Scoring**: Scores candidate words using crossing letter compatibility and character frequencies.
3. **Dynamic Graph Decomposition**: Disconnects independent grid components and solves them sequentially.
