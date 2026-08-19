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
    ) -> Option<CrosswordCore> {
        let effective_word_list = if let Some(thresh) = score_filter {
            Arc::new(word_list.score_filter(thresh))
        } else {
            Arc::clone(word_list)
        };

        let mut cloned = self.clone();
        let subgrids = cloned.grid.get_open_subgrids();
        let start_time = Instant::now();
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
                &mut dead_end_states,
                &mut used_words,
            ) {
                return None;
            }
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
        dead_end_states: &mut AHashSet<u64>,
        used_words: &mut AHashSet<String>,
    ) -> bool {
        // Check timeout
        if let Some(timeout) = timeout_secs {
            if start_time.elapsed().as_secs_f64() > timeout {
                return false;
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
                                weights[i] = cross_letter_scores[i] * INVERSE_CHARACTER_FREQUENCIES[i];
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

        // 7. Check if placing a word splits the subgraph into smaller components
        let first_word_bytes = part.get_word_bytes(candidate_indices[0]);
        for (i, &(r, c)) in slot.cell_indices.iter().enumerate() {
            xw.grid.get_cell_mut(r, c).value = CellValue::Letter(first_word_bytes[i] as char);
        }
        let new_subgraphs: Vec<Vec<WordIndex>> = xw
            .grid
            .get_open_subgrids()
            .into_iter()
            .filter(|sub| sub.iter().all(|w| active_subgraph.contains(w)))
            .collect();

        // 8. Try candidate words
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

            let mut all_solved = true;
            for new_sub in &new_subgraphs {
                if !Self::solve_subgraph(
                    xw,
                    new_sub,
                    word_list,
                    start_time,
                    timeout_secs,
                    temperature,
                    allow_repeats,
                    dead_end_states,
                    used_words,
                ) {
                    all_solved = false;
                    break;
                }
            }

            if all_solved {
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
}

// Simple fast pseudo-random generator for heuristic noise
static mut SEED: u64 = 0x853c49e6748fea9b;
fn rand_simple() -> f64 {
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((SEED >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
