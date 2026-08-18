# Word Lists & Scoring

`blacksquare` includes a high-performance word list engine capable of matching patterns and calculating crossing letter frequencies in microseconds.

---

## Embedded Default Word List

The default word list contains **304,661 scored English words** (from the open SpreadTheWordlist corpus) compiled directly into the binary:

```python
from blacksquare import DEFAULT_WORDLIST

print(f"Total words: {len(DEFAULT_WORDLIST):,}")
print("Top sample words:", DEFAULT_WORDLIST.sample(5))
```

---

## Custom Word Lists

You can construct a `WordList` from Python lists, dictionaries, or text files:

=== "From Python List"

    ```python
    from blacksquare import WordList

    wl = WordList(["CAT", "DOG", "BIRD", "FISH"])
    ```

=== "From Dict with Scores"

    ```python
    # Scores are scaled automatically between 0.0 and 1.0
    wl = WordList(
        {
            "HELLO": 100,
            "WORLD": 80,
            "CROSSWORD": 95,
        }
    )
    ```

=== "From Text File (`word;score`)"

    ```python
    # Reads lines in 'WORD;SCORE' format
    wl = WordList("custom_words.dict")
    ```

---

## Pattern Matching

Find matches for words with wildcards (`?`, ` `, `_`):

```python
matches = DEFAULT_WORDLIST.find_matches("C??S")
print(matches.words[:10])  # ['CATS', 'COWS', 'CARS', ...]
```

---

## Filtering and Combining

```python
# Filter words with score >= 0.8
high_quality_wl = DEFAULT_WORDLIST.score_filter(0.8)

# Combine two word lists (scores are merged)
combined_wl = word_list_1 + word_list_2
```
