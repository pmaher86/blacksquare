use ahash::{AHashMap, AHashSet};
use crate::cell::Cell;
use crate::types::{CellIndex, Direction, WordIndex};

#[derive(Clone, Debug)]
pub struct WordSlot {
    pub direction: Direction,
    pub number: u16,
    pub start_row: usize,
    pub start_col: usize,
    pub length: usize,
    pub cell_indices: Vec<CellIndex>,
    pub crosses: Vec<(Option<WordIndex>, usize)>,
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub num_rows: usize,
    pub num_cols: usize,
    pub cells: Vec<Cell>,
    pub numbers: Vec<u16>,
    pub across_numbers: Vec<u16>,
    pub down_numbers: Vec<u16>,
    pub word_slots: AHashMap<WordIndex, WordSlot>,
}

impl Grid {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        let size = num_rows * num_cols;
        let mut grid = Grid {
            num_rows,
            num_cols,
            cells: vec![Cell::empty(); size],
            numbers: vec![0; size],
            across_numbers: vec![0; size],
            down_numbers: vec![0; size],
            word_slots: AHashMap::new(),
        };
        grid.parse_grid();
        grid
    }

    pub fn from_cells(num_rows: usize, num_cols: usize, cells: Vec<Cell>) -> Self {
        assert_eq!(cells.len(), num_rows * num_cols);
        let mut grid = Grid {
            num_rows,
            num_cols,
            cells,
            numbers: vec![0; num_rows * num_cols],
            across_numbers: vec![0; num_rows * num_cols],
            down_numbers: vec![0; num_rows * num_cols],
            word_slots: AHashMap::new(),
        };
        grid.parse_grid();
        grid
    }

    #[inline(always)]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.num_cols + col
    }

    #[inline(always)]
    pub fn get_cell(&self, row: usize, col: usize) -> &Cell {
        &self.cells[self.idx(row, col)]
    }

    #[inline(always)]
    pub fn get_cell_mut(&mut self, row: usize, col: usize) -> &mut Cell {
        let i = self.idx(row, col);
        &mut self.cells[i]
    }

    pub fn is_black(&self, row: usize, col: usize) -> bool {
        if row >= self.num_rows || col >= self.num_cols {
            true
        } else {
            self.get_cell(row, col).is_black()
        }
    }

    pub fn parse_grid(&mut self) {
        let rows = self.num_rows;
        let cols = self.num_cols;
        let size = rows * cols;

        self.numbers.fill(0);
        self.across_numbers.fill(0);
        self.down_numbers.fill(0);
        self.word_slots.clear();

        let mut current_num: u16 = 1;
        let mut across_starts = vec![false; size];
        let mut down_starts = vec![false; size];

        // 1. Identify which cells start Across and Down words
        for r in 0..rows {
            for c in 0..cols {
                if self.is_black(r, c) {
                    continue;
                }

                // Check Across: left is black/boundary, right is in-bounds and not black
                let left_black = c == 0 || self.is_black(r, c - 1);
                let right_open = c + 1 < cols && !self.is_black(r, c + 1);
                let starts_across = left_black && right_open;

                // Check Down: top is black/boundary, bottom is in-bounds and not black
                let top_black = r == 0 || self.is_black(r - 1, c);
                let bottom_open = r + 1 < rows && !self.is_black(r + 1, c);
                let starts_down = top_black && bottom_open;

                let i = self.idx(r, c);
                across_starts[i] = starts_across;
                down_starts[i] = starts_down;

                if starts_across || starts_down {
                    self.numbers[i] = current_num;
                    current_num += 1;
                }
            }
        }

        // 2. Propagate across numbers and build Across word slots
        for r in 0..rows {
            let mut c = 0;
            while c < cols {
                if !self.is_black(r, c) && across_starts[self.idx(r, c)] {
                    let num = self.numbers[self.idx(r, c)];
                    let start_col = c;
                    let mut cell_indices = Vec::new();

                    while c < cols && !self.is_black(r, c) {
                        let i = self.idx(r, c);
                        self.across_numbers[i] = num;
                        cell_indices.push((r, c));
                        c += 1;
                    }

                    let length = cell_indices.len();
                    self.word_slots.insert(
                        (Direction::Across, num),
                        WordSlot {
                            direction: Direction::Across,
                            number: num,
                            start_row: r,
                            start_col,
                            length,
                            cell_indices,
                            crosses: Vec::new(),
                        },
                    );
                } else {
                    c += 1;
                }
            }
        }

        // 3. Propagate down numbers and build Down word slots
        for c in 0..cols {
            let mut r = 0;
            while r < rows {
                if !self.is_black(r, c) && down_starts[self.idx(r, c)] {
                    let num = self.numbers[self.idx(r, c)];
                    let start_row = r;
                    let mut cell_indices = Vec::new();

                    while r < rows && !self.is_black(r, c) {
                        let i = self.idx(r, c);
                        self.down_numbers[i] = num;
                        cell_indices.push((r, c));
                        r += 1;
                    }

                    let length = cell_indices.len();
                    self.word_slots.insert(
                        (Direction::Down, num),
                        WordSlot {
                            direction: Direction::Down,
                            number: num,
                            start_row,
                            start_col: c,
                            length,
                            cell_indices,
                            crosses: Vec::new(),
                        },
                    );
                } else {
                    r += 1;
                }
            }
        }

        // 4. Precompute cross words for each slot
        let across_nums = self.across_numbers.clone();
        let down_nums = self.down_numbers.clone();

        // Collect word keys
        let word_keys: Vec<WordIndex> = self.word_slots.keys().copied().collect();

        for wi in word_keys {
            let slot = self.word_slots.get(&wi).unwrap();
            let mut crosses = Vec::with_capacity(slot.length);

            for &(r, c) in &slot.cell_indices {
                let i = r * cols + c;
                match slot.direction {
                    Direction::Across => {
                        let dn = down_nums[i];
                        if dn > 0 {
                            if let Some(down_slot) = self.word_slots.get(&(Direction::Down, dn)) {
                                let offset = down_slot
                                    .cell_indices
                                    .iter()
                                    .position(|&coords| coords == (r, c))
                                    .unwrap_or(0);
                                crosses.push((Some((Direction::Down, dn)), offset));
                            } else {
                                crosses.push((None, 0));
                            }
                        } else {
                            crosses.push((None, 0));
                        }
                    }
                    Direction::Down => {
                        let an = across_nums[i];
                        if an > 0 {
                            if let Some(across_slot) = self.word_slots.get(&(Direction::Across, an)) {
                                let offset = across_slot
                                    .cell_indices
                                    .iter()
                                    .position(|&coords| coords == (r, c))
                                    .unwrap_or(0);
                                crosses.push((Some((Direction::Across, an)), offset));
                            } else {
                                crosses.push((None, 0));
                            }
                        } else {
                            crosses.push((None, 0));
                        }
                    }
                }
            }

            self.word_slots.get_mut(&wi).unwrap().crosses = crosses;
        }
    }

    pub fn get_word_value(&self, word_index: WordIndex) -> Option<String> {
        let slot = self.word_slots.get(&word_index)?;
        let mut s = String::with_capacity(slot.length);
        for &(r, c) in &slot.cell_indices {
            s.push_str(&self.get_cell(r, c).value.to_str());
        }
        Some(s)
    }

    pub fn is_word_open(&self, word_index: WordIndex) -> bool {
        if let Some(slot) = self.word_slots.get(&word_index) {
            slot.cell_indices.iter().any(|&(r, c)| self.get_cell(r, c).is_open())
        } else {
            false
        }
    }

    pub fn get_word_at_cell(&self, row: usize, col: usize, direction: Direction) -> Option<WordIndex> {
        let i = self.idx(row, col);
        let num = match direction {
            Direction::Across => self.across_numbers[i],
            Direction::Down => self.down_numbers[i],
        };
        if num > 0 {
            Some((direction, num))
        } else {
            None
        }
    }

    /// Finds connected components of open words that share open cells.
    pub fn get_open_subgrids(&self) -> Vec<Vec<WordIndex>> {
        let mut open_words: Vec<WordIndex> = self
            .word_slots
            .keys()
            .copied()
            .filter(|&w| self.is_word_open(w))
            .collect();
        open_words.sort();

        let mut visited = AHashSet::new();
        let mut subgrids = Vec::new();

        for &start_word in &open_words {
            if visited.contains(&start_word) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = vec![start_word];
            visited.insert(start_word);

            while let Some(w) = queue.pop() {
                component.push(w);
                if let Some(slot) = self.word_slots.get(&w) {
                    for (idx, &(r, c)) in slot.cell_indices.iter().enumerate() {
                        if self.get_cell(r, c).is_open() {
                            if let Some((Some(cross_w), _)) = slot.crosses.get(idx) {
                                if self.is_word_open(*cross_w) && visited.insert(*cross_w) {
                                    queue.push(*cross_w);
                                }
                            }
                        }
                    }
                }
            }

            component.sort();
            subgrids.push(component);
        }

        subgrids.sort_by_key(|s| s.len());
        subgrids
    }

    /// Fast connected component decomposition scoped strictly to active_subgraph.
    pub fn get_subgraph_components(&self, active_subgraph: &[WordIndex]) -> Vec<Vec<WordIndex>> {
        let open_words: Vec<WordIndex> = active_subgraph
            .iter()
            .copied()
            .filter(|&w| self.is_word_open(w))
            .collect();

        if open_words.len() <= 1 {
            if open_words.is_empty() {
                return Vec::new();
            } else {
                return vec![open_words];
            }
        }

        let open_set: AHashSet<WordIndex> = open_words.iter().copied().collect();
        let mut visited = AHashSet::with_capacity(open_words.len());
        let mut subgrids = Vec::with_capacity(2);

        for &start_word in &open_words {
            if visited.contains(&start_word) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = vec![start_word];
            visited.insert(start_word);

            while let Some(w) = queue.pop() {
                component.push(w);
                if let Some(slot) = self.word_slots.get(&w) {
                    for (idx, &(r, c)) in slot.cell_indices.iter().enumerate() {
                        if self.get_cell(r, c).is_open() {
                            if let Some((Some(cross_w), _)) = slot.crosses.get(idx) {
                                if open_set.contains(cross_w) && visited.insert(*cross_w) {
                                    queue.push(*cross_w);
                                }
                            }
                        }
                    }
                }
            }

            // Quick check: if the first component contains all open words, no split!
            if component.len() == open_words.len() {
                return vec![component];
            }

            component.sort();
            subgrids.push(component);
        }

        subgrids.sort_by_key(|s| s.len());
        subgrids
    }
}
