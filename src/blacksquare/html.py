CSS_TEMPLATE = """
.crossword{suffix} {{
    display: grid;
    grid-template-columns: repeat({num_cols}, 1fr);
    grid-auto-rows: 1fr;
    gap: 0px;
    width: {width}px;
    height: {height}px;
}}

.crossword-cell{suffix} {{
    position: relative;
    background-color: white;
    outline: 1px solid black;
    font-family: Arial, Helvetica, sans-serif;
    aspect-ratio: 1;
}}

.crossword-cell{suffix} .cell-number {{
    position: absolute;
    top: 2px;
    left: 2px;
    font-size: {num_font_size}px;
    color: black;
    user-select: none;
}}

.crossword-cell{suffix} .letter {{
    font-size: {val_font_size}px;
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translate(-50%, 0%);
    color: black;
}}

.black {{
    background-color: black;
    outline: 1px solid gray;
}}

.gray {{
    background-color: lightgray;
}}

.crossword-cell{suffix} .circle {{
    position: absolute;
    border-radius: 50%;
    border: 1px solid black;
    height: {circle_dim}px;
    width: {circle_dim}px;
    margin: -1px;
}}
"""
