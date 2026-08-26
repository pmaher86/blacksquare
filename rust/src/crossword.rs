use ahash::{AHashMap, AHashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::cell::{Cell, CellValue};
use crate::grid::Grid;
use crate::symmetry::Symmetry;
use crate::types::{CellIndex, Direction, WordIndex};
use crate::word_list::{FastWordList, INVERSE_CHARACTER_FREQUENCIES};

#[derive(Clone, Debug)]
pub struct CrosswordCore {
    pub grid: Grid,
    pub clues: AHashMap<WordIndex, String>,
    pub symmetry: Option<Symmetry>,
    pub display_size_px: u32,
}

impl CrosswordCore {
    pub fn new(num_rows: usize, num_cols: usize, symmetry: Option<Symmetry>, display_size_px: u32) -> Self {
        CrosswordCore {
            grid: Grid::new(num_rows, num_cols),
            clues: AHashMap::new(),
            symmetry,
            display_size_px,
        }
    }

    /// Formats the crossword grid as a formatted text table matching the classic continuous square box style.
    pub fn to_text_grid(&self, numbers: bool) -> String {
        const SUPERSCRIPT_DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
        let num_rows = self.grid.num_rows;
        let num_cols = self.grid.num_cols;
        if num_rows == 0 || num_cols == 0 {
            return String::new();
        }

        let mut out = String::with_capacity(num_rows * num_cols * 10);

        // 1. Top border: ┌───┬───┬───┐
        out.push('┌');
        for c in 0..num_cols {
            out.push_str("───");
            if c < num_cols - 1 {
                out.push('┬');
            }
        }
        out.push_str("┐\n");

        // 2. Rows
        for r in 0..num_rows {
            out.push('│');
            for c in 0..num_cols {
                let cell = self.grid.get_cell(r, c);
                let num = self.grid.numbers[self.grid.idx(r, c)];
                match &cell.value {
                    CellValue::Black => {
                        out.push_str("███");
                    }
                    CellValue::Letter(ch) => {
                        let prefix = if num > 0 {
                            SUPERSCRIPT_DIGITS[(num % 10) as usize]
                        } else {
                            ' '
                        };
                        let suffix = if cell.shaded || cell.circled { '*' } else { ' ' };
                        out.push(prefix);
                        out.push(*ch);
                        out.push(suffix);
                    }
                    CellValue::Rebus { across, down } => {
                        let prefix = if num > 0 {
                            SUPERSCRIPT_DIGITS[(num % 10) as usize]
                        } else {
                            ' '
                        };
                        let suffix = if cell.shaded || cell.circled { '*' } else { ' ' };
                        let val = if across == down { across } else { across };
                        out.push(prefix);
                        out.push_str(val);
                        out.push(suffix);
                    }
                    CellValue::Schrodinger(parts) => {
                        let prefix = if num > 0 {
                            SUPERSCRIPT_DIGITS[(num % 10) as usize]
                        } else {
                            ' '
                        };
                        let suffix = if cell.shaded || cell.circled { '*' } else { ' ' };
                        let val = parts.join("/");
                        out.push(prefix);
                        out.push_str(&val);
                        out.push(suffix);
                    }
                    CellValue::Empty => {
                        if numbers && num > 0 {
                            let s = format!("{:^3}", num);
                            out.push_str(&s);
                        } else if num > 0 {
                            let prefix = SUPERSCRIPT_DIGITS[(num % 10) as usize];
                            let suffix = if cell.shaded || cell.circled { '*' } else { ' ' };
                            out.push(prefix);
                            out.push(' ');
                            out.push(suffix);
                        } else {
                            let suffix = if cell.shaded || cell.circled { '*' } else { ' ' };
                            out.push(' ');
                            out.push(' ');
                            out.push(suffix);
                        }
                    }
                }
                out.push('│');
            }
            out.push('\n');

            // Inter-row divider: ├───┼───┼───┤
            if r < num_rows - 1 {
                out.push('├');
                for c in 0..num_cols {
                    out.push_str("───");
                    if c < num_cols - 1 {
                        out.push('┼');
                    }
                }
                out.push_str("┤\n");
            }
        }

        // 3. Bottom border: └───┴───┴───┘
        out.push('└');
        for c in 0..num_cols {
            out.push_str("───");
            if c < num_cols - 1 {
                out.push('┴');
            }
        }
        out.push('┘');

        out
    }

    pub fn from_cells(num_rows: usize, num_cols: usize, cells: Vec<Cell>, symmetry: Option<Symmetry>, display_size_px: u32) -> Self {
        CrosswordCore {
            grid: Grid::from_cells(num_rows, num_cols, cells),
            clues: AHashMap::new(),
            symmetry,
            display_size_px,
        }
    }

    pub fn reparse_and_preserve_clues(&mut self) {
        let mut old_across_clues: AHashMap<Vec<CellIndex>, String> = AHashMap::new();
        let mut old_down_clues: AHashMap<Vec<CellIndex>, String> = AHashMap::new();

        for (&(dir, num), slot) in &self.grid.word_slots {
            if let Some(clue) = self.clues.get(&(dir, num)) {
                if !clue.is_empty() {
                    match dir {
                        Direction::Across => {
                            old_across_clues.insert(slot.cell_indices.clone(), clue.clone());
                        }
                        Direction::Down => {
                            old_down_clues.insert(slot.cell_indices.clone(), clue.clone());
                        }
                    }
                }
            }
        }

        self.grid.parse_grid();

        let mut new_clues: AHashMap<WordIndex, String> = AHashMap::new();
        for (&(dir, num), slot) in &self.grid.word_slots {
            let matched_clue = match dir {
                Direction::Across => old_across_clues.get(&slot.cell_indices),
                Direction::Down => old_down_clues.get(&slot.cell_indices),
            };
            if let Some(clue) = matched_clue {
                new_clues.insert((dir, num), clue.clone());
            }
        }
        self.clues = new_clues;
    }

    pub fn set_cell(&mut self, row: usize, col: usize, value: CellValue) {
        let is_currently_black = self.grid.get_cell(row, col).is_black();
        let is_new_black = value.is_black();

        if is_new_black {
            self.grid.get_cell_mut(row, col).value = CellValue::Black;
            if let Some(sym) = self.symmetry {
                let images = sym.apply_cell(row, col, self.grid.num_rows, self.grid.num_cols);
                for img in images {
                    let (r, c) = img.cell_index;
                    self.grid.get_cell_mut(r, c).value = CellValue::Black;
                }
            }
            self.reparse_and_preserve_clues();
        } else if is_currently_black {
            self.grid.get_cell_mut(row, col).value = value;
            if let Some(sym) = self.symmetry {
                let images = sym.apply_cell(row, col, self.grid.num_rows, self.grid.num_cols);
                for img in images {
                    let (r, c) = img.cell_index;
                    if self.grid.get_cell(r, c).is_black() {
                        self.grid.get_cell_mut(r, c).value = CellValue::Empty;
                    }
                }
            }
            self.reparse_and_preserve_clues();
        } else {
            self.grid.get_cell_mut(row, col).value = value;
        }
    }

    pub fn set_cell_rebus(&mut self, row: usize, col: usize, across: String, down: String) {
        let is_currently_black = self.grid.get_cell(row, col).is_black();
        let value = CellValue::Rebus { across, down };
        if is_currently_black {
            self.grid.get_cell_mut(row, col).value = value;
            if let Some(sym) = self.symmetry {
                let images = sym.apply_cell(row, col, self.grid.num_rows, self.grid.num_cols);
                for img in images {
                    let (r, c) = img.cell_index;
                    if self.grid.get_cell(r, c).is_black() {
                        self.grid.get_cell_mut(r, c).value = CellValue::Empty;
                    }
                }
            }
            self.reparse_and_preserve_clues();
        } else {
            self.grid.get_cell_mut(row, col).value = value;
        }
    }

    pub fn set_word(&mut self, word_index: WordIndex, value: &str) -> Result<(), String> {
        let slot = self
            .grid
            .word_slots
            .get(&word_index)
            .ok_or_else(|| format!("Word {:?} not found in grid", word_index))?
            .clone();

        let chars: Vec<char> = value.chars().collect();
        if chars.len() != slot.length {
            return Err(format!(
                "Value length {} does not match word length {}",
                chars.len(),
                slot.length
            ));
        }

        for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
            let cell_val = CellValue::parse(&chars[i].to_string())?;
            self.grid.get_cell_mut(r, c).value = cell_val;
        }
        Ok(())
    }

    pub fn get_symmetric_cell_indices(&self, row: usize, col: usize) -> Vec<(usize, usize)> {
        if let Some(sym) = self.symmetry {
            sym.apply_cell(row, col, self.grid.num_rows, self.grid.num_cols)
                .into_iter()
                .map(|img| img.cell_index)
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_symmetric_word_indices(&self, word_index: WordIndex) -> Vec<WordIndex> {
        let sym = match self.symmetry {
            Some(s) => s,
            None => return Vec::new(),
        };

        let slot = match self.grid.word_slots.get(&word_index) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let &(first_r, first_c) = match slot.cell_indices.first() {
            Some(coords) => coords,
            None => return Vec::new(),
        };

        let images = sym.apply_cell(first_r, first_c, self.grid.num_rows, self.grid.num_cols);
        let mut result = Vec::new();

        for img in images {
            let (r, c) = img.cell_index;
            let target_dir = if img.word_direction_rotated {
                slot.direction.opposite()
            } else {
                slot.direction
            };

            if let Some(w) = self.grid.get_word_at_cell(r, c, target_dir) {
                if !result.contains(&w) {
                    result.push(w);
                }
            }
        }
        result
    }

    /// Computes a fast 64-bit state hash for dead-end caching over a subgraph.
    #[inline]
    pub fn compute_subgraph_hash(&self, subgraph: &[WordIndex]) -> u64 {
        let mut hasher = ahash::AHasher::default();
        use std::hash::Hasher;

        for &wi in subgraph {
            if let Some(slot) = self.grid.word_slots.get(&wi) {
                hasher.write_u8(slot.direction as u8);
                hasher.write_u16(slot.number);
                for &(r, c) in &slot.cell_indices {
                    let ch = self.grid.get_cell(r, c).value.char_or_wildcard();
                    hasher.write_u8(ch as u8);
                }
            }
        }
        hasher.finish()
    }

    /// High-performance sequential backtracking solver.
    pub fn fill(
        &self,
        word_list: &Arc<FastWordList>,
        timeout_secs: Option<f64>,
        temperature: f64,
        score_filter: Option<f64>,
        allow_repeats: bool,
        upweight_diverse_letters: bool,
        show_progress: bool,
    ) -> Option<CrosswordCore> {
        let effective_word_list = if let Some(thresh) = score_filter {
            Arc::new(word_list.score_filter(thresh))
        } else {
            Arc::clone(word_list)
        };

        let mut cloned = self.clone();
        let subgrids = cloned.grid.get_open_subgrids();
        let start_time = Instant::now();
        let mut last_display = Instant::now();
        let mut states_visited: usize = 0;
        let mut displayed_lines: usize = 0;
        let mut dead_end_states: AHashSet<u64> = AHashSet::new();

        // Persistent used_words set on recursion stack
        let mut used_words: AHashSet<String> = AHashSet::new();
        if !allow_repeats {
            for &w in cloned.grid.word_slots.keys() {
                if !cloned.grid.is_word_open(w) {
                    if let Some(val) = cloned.grid.get_word_value(w) {
                        used_words.insert(val);
                    }
                }
            }
        }

        for subgraph in subgrids {
            if !Self::solve_subgraph(
                &mut cloned,
                &subgraph,
                &effective_word_list,
                &start_time,
                timeout_secs,
                temperature,
                allow_repeats,
                upweight_diverse_letters,
                show_progress,
                &mut last_display,
                &mut states_visited,
                &mut displayed_lines,
                &mut dead_end_states,
                &mut used_words,
            ) {
                if displayed_lines > 0 {
                    let grid_str = cloned.to_text_grid(false);
                    let exhaust_frame = format!(
                        "=== Crossword Search Exhausted (Time: {:.3}s, States: {}) ===\n{}\n",
                        start_time.elapsed().as_secs_f64(),
                        states_visited,
                        grid_str
                    );
                    eprint!("\x1b[{}A\r\x1b[J{}\x1b[?25h", displayed_lines, exhaust_frame);
                }
                return None;
            }
        }

        if displayed_lines > 0 {
            let grid_str = cloned.to_text_grid(false);
            let final_frame = format!(
                "=== Crossword Solved in {:.3}s ({} states explored) ===\n{}\n",
                start_time.elapsed().as_secs_f64(),
                states_visited,
                grid_str
            );
            eprint!("\x1b[{}A\r\x1b[J{}\x1b[?25h", displayed_lines, final_frame);
        }

        Some(cloned)
    }

    fn solve_subgraph(
        xw: &mut CrosswordCore,
        active_subgraph: &[WordIndex],
        word_list: &Arc<FastWordList>,
        start_time: &Instant,
        timeout_secs: Option<f64>,
        temperature: f64,
        allow_repeats: bool,
        upweight_diverse_letters: bool,
        show_progress: bool,
        last_display: &mut Instant,
        states_visited: &mut usize,
        displayed_lines: &mut usize,
        dead_end_states: &mut AHashSet<u64>,
        used_words: &mut AHashSet<String>,
    ) -> bool {
        // Check timeout
        if let Some(timeout) = timeout_secs {
            if start_time.elapsed().as_secs_f64() > timeout {
                return false;
            }
        }

        *states_visited += 1;
        if show_progress && start_time.elapsed() >= std::time::Duration::from_millis(100) {
            if last_display.elapsed() >= std::time::Duration::from_millis(100) {
                *last_display = Instant::now();
                let grid_str = xw.to_text_grid(false);
                let frame = format!(
                    "=== Crossword Fill in Progress [Elapsed: {:.2}s | States: {}] ===\n{}\n",
                    start_time.elapsed().as_secs_f64(),
                    *states_visited,
                    grid_str
                );
                let new_lines = frame.lines().count();
                if *displayed_lines > 0 {
                    eprint!("\x1b[{}A\r\x1b[J", *displayed_lines);
                } else {
                    eprint!("\x1b[?25l"); // hide cursor on first render
                }
                eprint!("{}", frame);
                *displayed_lines = new_lines;
            }
        }

        // 1. Check dead-end cache using optimistic full-dictionary hash
        let state_hash = xw.compute_subgraph_hash(active_subgraph);
        if dead_end_states.contains(&state_hash) {
            return false;
        }

        // 2. Filter open words in this subgraph
        let mut open_words: Vec<WordIndex> = active_subgraph
            .iter()
            .copied()
            .filter(|&w| xw.grid.is_word_open(w))
            .collect();
        open_words.sort();

        if open_words.is_empty() {
            return true; // Subgraph fully solved!
        }

        // 3. Minimum Remaining Values (MRV) Variable Ordering
        let mut best_word: Option<WordIndex> = None;
        let mut min_matches = usize::MAX;
        let mut best_mask = Vec::new();
        let mut temp_mask = Vec::new();

        for &w in &open_words {
            let slot = match xw.grid.word_slots.get(&w) {
                Some(s) => s,
                None => continue,
            };

            let part = match &word_list.partitions[slot.length] {
                Some(p) => p,
                None => {
                    dead_end_states.insert(state_hash);
                    return false;
                }
            };

            // Build query pattern in stack buffer (zero heap allocation)
            let mut pattern = [b'?'; 32];
            for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
                let ch = xw.grid.get_cell(r, c).value.char_or_wildcard();
                pattern[i] = if ch == '?' || ch == ' ' { b'?' } else { ch as u8 };
            }

            temp_mask.clear();
            part.match_pattern(&pattern[..slot.length], &mut temp_mask);
            let count: usize = temp_mask.iter().map(|w| w.count_ones() as usize).sum();

            if count == 0 {
                // Immediate domain wipeout: dead end!
                dead_end_states.insert(state_hash);
                return false;
            }

            let scored_count = if temperature > 0.0 {
                let noise = (rand_simple() * temperature * count as f64).abs() as usize;
                count + noise
            } else {
                count
            };

            if scored_count < min_matches {
                min_matches = scored_count;
                best_word = Some(w);
                best_mask.clear();
                best_mask.extend_from_slice(&temp_mask);
            }
        }

        let word_to_match = match best_word {
            Some(w) => w,
            None => return true,
        };
        let slot = xw.grid.word_slots.get(&word_to_match).unwrap().clone();
        let part = word_list.partitions[slot.length].as_ref().unwrap();

        // 4. Compute cross scoring weights for candidate ranking
        let mut open_positions = Vec::new();
        let mut letter_weights_per_pos = Vec::new();
        let mut crossing_slots_info = Vec::new();

        for (idx, &(r, c)) in slot.cell_indices.iter().enumerate() {
            if xw.grid.get_cell(r, c).is_open() {
                if let Some((Some(cross_w), cross_offset)) = slot.crosses.get(idx) {
                    if let Some(cross_slot) = xw.grid.word_slots.get(cross_w) {
                        crossing_slots_info.push((*cross_w, *cross_offset, cross_slot.clone()));

                        let mut cross_pattern = [b'?'; 32];
                        for (i, &(cr, cc)) in cross_slot.cell_indices.iter().enumerate() {
                            let ch = xw.grid.get_cell(cr, cc).value.char_or_wildcard();
                            cross_pattern[i] = if ch == '?' || ch == ' ' { b'?' } else { ch as u8 };
                        }

                        if let Some(cross_part) = &word_list.partitions[cross_slot.length] {
                            let mut cross_mask = Vec::new();
                            cross_part.match_pattern(&cross_pattern[..cross_slot.length], &mut cross_mask);
                            let cross_letter_scores = cross_part.letter_scores_at_index(&cross_mask, *cross_offset);

                            let mut weights = [0.0f64; 26];
                            for i in 0..26 {
                                if upweight_diverse_letters {
                                    weights[i] = cross_letter_scores[i]
                                        * INVERSE_CHARACTER_FREQUENCIES[i];
                                } else {
                                    weights[i] = cross_letter_scores[i];
                                }
                            }
                            open_positions.push(idx);
                            letter_weights_per_pos.push(weights);
                        }
                    }
                }
            }
        }

        // 5. Zero-allocation candidate indices extraction
        let candidate_indices: Vec<usize> = if !open_positions.is_empty() {
            part.fused_cross_score_indices(&best_mask, &open_positions, &letter_weights_per_pos, true)
        } else {
            let mut indices = Vec::new();
            for (chunk_idx, &bits) in best_mask.iter().enumerate() {
                let mut b = bits;
                while b != 0 {
                    let bit_pos = b.trailing_zeros() as usize;
                    let word_idx = chunk_idx * 64 + bit_pos;
                    if word_idx < part.count {
                        indices.push(word_idx);
                    }
                    b &= b - 1;
                }
            }
            indices
        };

        if candidate_indices.is_empty() {
            dead_end_states.insert(state_hash);
            return false;
        }

        // 6. Save old cell values for fast backtracking (undo stack)
        let old_values: Vec<CellValue> = slot
            .cell_indices
            .iter()
            .map(|&(r, c)| xw.grid.get_cell(r, c).value.clone())
            .collect();

        // 7. Try candidate words
        'candidate_loop: for &word_idx in &candidate_indices {
            let word_bytes = part.get_word_bytes(word_idx);
            let word_str = part.get_word_str(word_idx);

            if !allow_repeats && used_words.contains(word_str) {
                continue;
            }

            if let Some(timeout) = timeout_secs {
                if start_time.elapsed().as_secs_f64() > timeout {
                    // Restore on timeout
                    for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
                        xw.grid.get_cell_mut(r, c).value = old_values[i].clone();
                    }
                    return false;
                }
            }

            for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
                xw.grid.get_cell_mut(r, c).value = CellValue::Letter(word_bytes[i] as char);
            }

            // Fast forward checking on crossing slots
            for (_cw, _off, cs) in &crossing_slots_info {
                let mut cp = [b'?'; 32];
                for (i, &(cr, cc)) in cs.cell_indices.iter().enumerate() {
                    let ch = xw.grid.get_cell(cr, cc).value.char_or_wildcard();
                    cp[i] = if ch == '?' || ch == ' ' { b'?' } else { ch as u8 };
                }
                if let Some(cs_part) = &word_list.partitions[cs.length] {
                    if !cs_part.has_match(&cp[..cs.length]) {
                        continue 'candidate_loop;
                    }
                } else {
                    continue 'candidate_loop;
                }
            }

            if !allow_repeats {
                used_words.insert(word_str.to_string());
            }

            if Self::solve_subgraph(
                xw,
                active_subgraph,
                word_list,
                start_time,
                timeout_secs,
                temperature,
                allow_repeats,
                upweight_diverse_letters,
                show_progress,
                last_display,
                states_visited,
                displayed_lines,
                dead_end_states,
                used_words,
            ) {
                return true;
            }

            if !allow_repeats {
                used_words.remove(word_str);
            }
        }

        // Backtrack: restore original values
        for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
            xw.grid.get_cell_mut(r, c).value = old_values[i].clone();
        }
        dead_end_states.insert(state_hash);
        false
    }

    pub fn check(
        &self,
        symmetry: Option<Symmetry>,
        min_word_length: usize,
        allow_duplicates: bool,
        require_connected: bool,
        require_filled: bool,
    ) -> (bool, Vec<String>, Vec<String>) {
        let mut errors: Vec<String> = Vec::new();
        let warnings: Vec<String> = Vec::new();

        let num_rows = self.grid.num_rows;
        let num_cols = self.grid.num_cols;

        // 1. Across segment length check
        for r in 0..num_rows {
            let mut col_start: Option<usize> = None;
            for c in 0..=num_cols {
                let is_white = c < num_cols && !self.grid.is_black(r, c);
                if is_white {
                    if col_start.is_none() {
                        col_start = Some(c);
                    }
                } else if let Some(start) = col_start {
                    let len = c - start;
                    if len < min_word_length {
                        errors.push(format!(
                            "Row {}, cols {}..{}: Across segment of length {} is shorter than minimum {}.",
                            r,
                            start,
                            c - 1,
                            len,
                            min_word_length
                        ));
                    }
                    col_start = None;
                }
            }
        }

        // 2. Down segment length check
        for c in 0..num_cols {
            let mut row_start: Option<usize> = None;
            for r in 0..=num_rows {
                let is_white = r < num_rows && !self.grid.is_black(r, c);
                if is_white {
                    if row_start.is_none() {
                        row_start = Some(r);
                    }
                } else if let Some(start) = row_start {
                    let len = r - start;
                    if len < min_word_length {
                        errors.push(format!(
                            "Col {}, rows {}..{}: Down segment of length {} is shorter than minimum {}.",
                            c,
                            start,
                            r - 1,
                            len,
                            min_word_length
                        ));
                    }
                    row_start = None;
                }
            }
        }

        // 3. Symmetry check
        let active_sym = symmetry.or(self.symmetry);
        if let Some(sym) = active_sym {
            let mut reported_cells: AHashSet<(usize, usize)> = AHashSet::new();
            for r in 0..num_rows {
                for c in 0..num_cols {
                    let is_black = self.grid.is_black(r, c);
                    let images = sym.apply_cell(r, c, num_rows, num_cols);
                    for img in images {
                        let (pr, pc) = img.cell_index;
                        if pr < num_rows && pc < num_cols {
                            let partner_is_black = self.grid.is_black(pr, pc);
                            if is_black != partner_is_black
                                && !reported_cells.contains(&(r, c))
                            {
                                reported_cells.insert((r, c));
                                errors.push(format!(
                                    "Symmetry violation ({}) at cell ({}, {}).",
                                    sym.value(),
                                    r,
                                    c
                                ));
                            }
                        }
                    }
                }
            }
        }

        // 4. Duplicate words check
        if !allow_duplicates {
            let mut word_occurrences: AHashMap<String, Vec<String>> =
                AHashMap::new();
            for (&(dir, num), _slot) in &self.grid.word_slots {
                if !self.grid.is_word_open((dir, num)) {
                    if let Some(val) = self.grid.get_word_value((dir, num)) {
                        let dir_str = match dir {
                            Direction::Across => "Across",
                            Direction::Down => "Down",
                        };
                        word_occurrences
                            .entry(val)
                            .or_default()
                            .push(format!("{} {}", dir_str, num));
                    }
                }
            }

            let mut sorted_keys: Vec<String> =
                word_occurrences.keys().cloned().collect();
            sorted_keys.sort();
            for val in sorted_keys {
                let mut locs = word_occurrences.remove(&val).unwrap();
                if locs.len() > 1 {
                    locs.sort();
                    errors.push(format!(
                        "Duplicate word '{}' reused at {}.",
                        val,
                        locs.join(", ")
                    ));
                }
            }
        }

        // 5. Grid connectivity check
        if require_connected {
            let mut open_cells: Vec<(usize, usize)> = Vec::new();
            for r in 0..num_rows {
                for c in 0..num_cols {
                    if !self.grid.is_black(r, c) {
                        open_cells.push((r, c));
                    }
                }
            }

            if !open_cells.is_empty() {
                let mut visited: AHashSet<(usize, usize)> =
                    AHashSet::with_capacity(open_cells.len());
                let mut queue = vec![open_cells[0]];
                visited.insert(open_cells[0]);

                while let Some((curr_r, curr_c)) = queue.pop() {
                    let neighbors = [
                        (curr_r.wrapping_sub(1), curr_c),
                        (curr_r + 1, curr_c),
                        (curr_r, curr_c.wrapping_sub(1)),
                        (curr_r, curr_c + 1),
                    ];
                    for (nr, nc) in neighbors {
                        if nr < num_rows
                            && nc < num_cols
                            && !self.grid.is_black(nr, nc)
                            && visited.insert((nr, nc))
                        {
                            queue.push((nr, nc));
                        }
                    }
                }

                if visited.len() != open_cells.len() {
                    errors.push(format!(
                        "Grid is not fully connected: {} open cell(s) are disconnected from the main grid.",
                        open_cells.len() - visited.len()
                    ));
                }
            }
        }

        // 6. Filled check
        if require_filled {
            let mut open_cell_count = 0;
            for r in 0..num_rows {
                for c in 0..num_cols {
                    if self.grid.get_cell(r, c).is_open() {
                        open_cell_count += 1;
                    }
                }
            }
            if open_cell_count > 0 {
                errors.push(format!(
                    "Grid contains {} empty / open cell(s).",
                    open_cell_count
                ));
            }
        }

        (errors.is_empty(), errors, warnings)
    }

    pub fn stats(&self) -> CrosswordStatsData {
        let mut across_words = 0;
        let mut down_words = 0;
        let mut filled_words = 0;
        let mut open_words = 0;
        let mut word_length_counter: AHashMap<usize, usize> = AHashMap::new();

        for (&(dir, num), slot) in &self.grid.word_slots {
            match dir {
                Direction::Across => across_words += 1,
                Direction::Down => down_words += 1,
            }
            if self.grid.is_word_open((dir, num)) {
                open_words += 1;
            } else {
                filled_words += 1;
            }
            *word_length_counter.entry(slot.length).or_insert(0) += 1;
        }

        let mut word_length_counts: Vec<(usize, usize)> =
            word_length_counter.into_iter().collect();
        word_length_counts.sort_by(|a, b| b.0.cmp(&a.0)); // sort by length descending

        let mut black_squares = 0;
        let mut open_cells = 0;
        let mut rebus_count = 0;
        let mut circled_count = 0;
        let mut shaded_count = 0;
        let mut letter_counter: AHashMap<String, usize> = AHashMap::new();

        for cell in &self.grid.cells {
            if cell.is_black() {
                black_squares += 1;
            } else {
                open_cells += 1;
                if cell.circled {
                    circled_count += 1;
                }
                if cell.shaded {
                    shaded_count += 1;
                }
                match &cell.value {
                    CellValue::Rebus { across, down } => {
                        rebus_count += 1;
                        if across == down {
                            *letter_counter
                                .entry(across.clone())
                                .or_insert(0) += 1;
                        } else {
                            *letter_counter
                                .entry(format!("{}/{}", across, down))
                                .or_insert(0) += 1;
                        }
                    }
                    CellValue::Letter(c) => {
                        *letter_counter.entry(c.to_string()).or_insert(0) += 1;
                    }
                    _ => {}
                }
            }
        }

        let mut letter_counts: Vec<(String, usize)> =
            letter_counter.into_iter().collect();
        letter_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))); // count desc, then key asc

        CrosswordStatsData {
            total_words: self.grid.word_slots.len(),
            across_words,
            down_words,
            filled_words,
            open_words,
            black_squares,
            total_cells: self.grid.num_rows * self.grid.num_cols,
            open_cells,
            word_length_counts,
            letter_counts,
            rebus_count,
            circled_count,
            shaded_count,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CrosswordStatsData {
    pub total_words: usize,
    pub across_words: usize,
    pub down_words: usize,
    pub filled_words: usize,
    pub open_words: usize,
    pub black_squares: usize,
    pub total_cells: usize,
    pub open_cells: usize,
    pub word_length_counts: Vec<(usize, usize)>,
    pub letter_counts: Vec<(String, usize)>,
    pub rebus_count: usize,
    pub circled_count: usize,
    pub shaded_count: usize,
}

// Simple fast pseudo-random generator for heuristic noise
static mut SEED: u64 = 0x853c49e6748fea9b;
fn rand_simple() -> f64 {
    unsafe {
        SEED = SEED
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((SEED >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_check_valid_grid() {
        let xw = CrosswordCore::new(5, 5, Some(Symmetry::Rotational), 450);
        let (is_valid, errors, _warnings) =
            xw.check(None, 3, false, true, false);
        assert!(is_valid);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_rust_check_short_word_segment() {
        let mut xw = CrosswordCore::new(5, 5, None, 450);
        xw.set_cell(0, 1, CellValue::Black);
        let (is_valid, errors, _warnings) =
            xw.check(None, 3, false, true, false);
        assert!(!is_valid);
        assert!(errors
            .iter()
            .any(|e| e.contains("is shorter than minimum 3")));
    }

    #[test]
    fn test_rust_check_symmetry_violation() {
        let mut xw = CrosswordCore::new(5, 5, None, 450);
        xw.grid.get_cell_mut(0, 0).value = CellValue::Black;
        let (is_valid, errors, _warnings) =
            xw.check(Some(Symmetry::Rotational), 3, false, true, false);
        assert!(!is_valid);
        assert!(errors.iter().any(|e| e.contains("Symmetry violation")));
    }

    #[test]
    fn test_rust_check_duplicate_words() {
        let mut xw = CrosswordCore::new(5, 5, None, 450);
        xw.set_word((Direction::Across, 1), "ALPHA").unwrap();
        xw.set_word((Direction::Across, 6), "ALPHA").unwrap();
        let (is_valid, errors, _warnings) =
            xw.check(None, 3, false, true, false);
        assert!(!is_valid);
        assert!(errors.iter().any(|e| e.contains("Duplicate word 'ALPHA'")));
    }

    #[test]
    fn test_rust_stats() {
        let mut xw = CrosswordCore::new(5, 5, Some(Symmetry::Rotational), 450);
        xw.set_cell(2, 2, CellValue::Black);
        xw.grid.get_cell_mut(1, 1).circled = true;
        xw.grid.get_cell_mut(3, 3).shaded = true;
        xw.set_cell_rebus(4, 4, "STAR".into(), "STAR".into());

        let stats = xw.stats();
        assert_eq!(stats.total_cells, 25);
        assert_eq!(stats.black_squares, 1);
        assert_eq!(stats.open_cells, 24);
        assert_eq!(stats.total_words, 12);
        assert_eq!(stats.across_words, 6);
        assert_eq!(stats.down_words, 6);
        assert_eq!(stats.rebus_count, 1);
        assert_eq!(stats.circled_count, 1);
        assert_eq!(stats.shaded_count, 1);
    }

    #[test]
    fn test_rust_fill_upweight_diverse_letters() {
        let wl = Arc::new(FastWordList::default());
        let xw = CrosswordCore::new(3, 3, None, 450);

        let filled_default = xw.fill(&wl, Some(5.0), 0.0, None, false, false, false);
        assert!(filled_default.is_some());

        let filled_upweighted = xw.fill(&wl, Some(5.0), 0.0, None, false, true, false);
        assert!(filled_upweighted.is_some());
    }

    #[test]
    fn test_rust_to_text_grid() {
        let mut xw = CrosswordCore::new(3, 3, None, 450);
        xw.set_cell(1, 1, CellValue::Black);
        xw.set_cell(0, 0, CellValue::Letter('A'));
        let text_grid = xw.to_text_grid(false);
        assert!(text_grid.contains("██"));
        assert!(text_grid.contains("A"));
    }
}
