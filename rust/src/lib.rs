use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

mod cell;
mod crossword;
mod grid;
mod symmetry;
mod types;
mod word_list;

use cell::{Cell, CellValue};
use crossword::CrosswordCore;
pub use symmetry::Symmetry;
pub use types::Direction;
use word_list::{FastWordList, MatchWordListCore, INVERSE_CHARACTER_FREQUENCIES};

#[pyclass(name = "PyMatchWordList")]
#[derive(Clone)]
pub struct PyMatchWordList {
    pub inner: MatchWordListCore,
}

#[pymethods]
impl PyMatchWordList {
    #[new]
    #[pyo3(signature = (word_length, words, scores))]
    pub fn new(word_length: usize, words: Vec<String>, scores: Vec<f64>) -> Self {
        PyMatchWordList {
            inner: MatchWordListCore::materialized(word_length, words, scores),
        }
    }

    #[getter]
    pub fn word_length(&self) -> usize {
        self.inner.word_length
    }

    #[getter]
    pub fn words(&self) -> Vec<String> {
        self.inner.words()
    }

    #[getter]
    pub fn scores(&self) -> Vec<f64> {
        self.inner.scores()
    }

    pub fn letter_scores_at_index<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let scores_array = self.inner.letter_scores_at_index(index);
        let dict = PyDict::new(py);
        for i in 0..26 {
            let s = scores_array[i];
            if s > 0.0 {
                let ch = ((b'A' + i as u8) as char).to_string();
                dict.set_item(ch, s)?;
            }
        }
        Ok(dict)
    }

    pub fn get_score(&self, word: &str) -> Option<f64> {
        self.inner.get_score(word)
    }

    pub fn score_filter(&self, threshold: f64) -> PyMatchWordList {
        PyMatchWordList {
            inner: self.inner.score_filter(threshold),
        }
    }

    pub fn filter_words(&self, words: Vec<String>) -> PyMatchWordList {
        PyMatchWordList {
            inner: self.inner.filter_words(&words),
        }
    }

    #[pyo3(signature = (rescore_fn, drop_zeros=true))]
    pub fn rescore(
        &self,
        py: Python<'_>,
        rescore_fn: PyObject,
        drop_zeros: bool,
    ) -> PyResult<PyMatchWordList> {
        let (words, scores) = self.inner.extract_words_and_scores();
        let mut scored: Vec<(String, f64)> = Vec::with_capacity(words.len());

        for (w, s) in words.into_iter().zip(scores.into_iter()) {
            let res = rescore_fn.call1(py, (w.clone(), s))?;
            let new_score: f64 = res.extract(py)?;
            if !drop_zeros || new_score > 0.0 {
                scored.push((w, new_score));
            }
        }

        // Sort descending by score, tie-breaking by word
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let out_words: Vec<String> = scored.iter().map(|(w, _)| w.clone()).collect();
        let out_scores: Vec<f64> = scored.iter().map(|(_, s)| *s).collect();

        Ok(PyMatchWordList {
            inner: MatchWordListCore::materialized(self.inner.word_length, out_words, out_scores),
        })
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __getitem__(&self, idx: isize) -> PyResult<(String, f64)> {
        let len = self.inner.len() as isize;
        let actual_idx = if idx < 0 { idx + len } else { idx };
        if actual_idx < 0 || actual_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        if let Some(item) = self.inner.get_item(actual_idx as usize) {
            Ok(item)
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ))
        }
    }

    pub fn __contains__(&self, word: &str) -> bool {
        self.inner.get_score(word).is_some()
    }
}

#[pyclass(name = "PyWordList")]
#[derive(Clone)]
pub struct PyWordList {
    pub inner: Arc<FastWordList>,
}

#[pymethods]
impl PyWordList {
    #[new]
    #[pyo3(signature = (source))]
    pub fn new(source: Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = source.downcast::<PyDict>() {
            let mut entries = Vec::with_capacity(dict.len());
            for (k, v) in dict.iter() {
                let word: String = k.extract()?;
                let score: f64 = v.extract()?;
                entries.push((word, score));
            }
            return Ok(PyWordList {
                inner: Arc::new(FastWordList::new(entries)),
            });
        }

        if let Ok(list) = source.downcast::<PyList>() {
            let mut entries = Vec::with_capacity(list.len());
            for item in list.iter() {
                let word: String = item.extract()?;
                entries.push((word, 1.0));
            }
            return Ok(PyWordList {
                inner: Arc::new(FastWordList::new(entries)),
            });
        }

        if let Ok(path_str) = source.extract::<String>() {
            let inner = FastWordList::from_dict_file(&path_str)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            return Ok(PyWordList {
                inner: Arc::new(inner),
            });
        }

        Err(pyo3::exceptions::PyValueError::new_err(
            "Input type not recognized for PyWordList",
        ))
    }

    #[staticmethod]
    pub fn default() -> Self {
        PyWordList {
            inner: FastWordList::default_embedded(),
        }
    }

    #[staticmethod]
    pub fn from_words_scores(words: Vec<String>, scores: Vec<f64>) -> Self {
        let entries: Vec<(String, f64)> = words.into_iter().zip(scores.into_iter()).collect();
        PyWordList {
            inner: Arc::new(FastWordList::new(entries)),
        }
    }

    pub fn find_matches_str(&self, query: &str) -> PyMatchWordList {
        PyMatchWordList {
            inner: self.inner.find_matches_str(query),
        }
    }

    #[getter]
    pub fn words(&self) -> Vec<String> {
        self.inner.words.clone()
    }

    #[getter]
    pub fn scores(&self) -> Vec<f64> {
        self.inner.scores.clone()
    }

    pub fn get_score(&self, word: &str) -> Option<f64> {
        self.inner.get_score(word)
    }

    pub fn contains(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    pub fn score_filter(&self, threshold: f64) -> PyWordList {
        PyWordList {
            inner: Arc::new(self.inner.score_filter(threshold)),
        }
    }

    pub fn get_partition(&self, length: usize) -> Option<(Vec<String>, Vec<f64>)> {
        self.inner.get_partition(length)
    }

    #[pyo3(signature = (word_pattern, open_positions, letter_weights_per_pos, drop_zeros=true))]
    pub fn fused_cross_matches(
        &self,
        word_pattern: &str,
        open_positions: Vec<usize>,
        letter_weights_per_pos: Vec<Vec<f64>>,
        drop_zeros: bool,
    ) -> PyResult<PyMatchWordList> {
        let clean_pattern = word_pattern.to_uppercase();
        let pattern_bytes: Vec<u8> = clean_pattern
            .bytes()
            .map(|b| match b {
                b'?' | b' ' | b'_' | b'-' => b'?',
                _ => b,
            })
            .collect();
        let len = pattern_bytes.len();

        if len == 0 || len >= self.inner.partitions.len() {
            return Ok(PyMatchWordList {
                inner: MatchWordListCore::empty(len),
            });
        }

        if let Some(part) = &self.inner.partitions[len] {
            let mut mask = Vec::new();
            part.match_pattern(&pattern_bytes, &mut mask);

            let mut weights_fixed: Vec<[f64; 26]> =
                Vec::with_capacity(letter_weights_per_pos.len());
            for w in letter_weights_per_pos {
                let mut arr = [0.0f64; 26];
                for i in 0..26.min(w.len()) {
                    arr[i] = w[i];
                }
                weights_fixed.push(arr);
            }

            let (words, scores) =
                part.fused_cross_score(&mask, &open_positions, &weights_fixed, drop_zeros);
            Ok(PyMatchWordList {
                inner: MatchWordListCore::materialized(len, words, scores),
            })
        } else {
            Ok(PyMatchWordList {
                inner: MatchWordListCore::empty(len),
            })
        }
    }

    pub fn __len__(&self) -> usize {
        self.inner.words.len()
    }

    pub fn __getitem__(&self, idx: isize) -> PyResult<(String, f64)> {
        let len = self.inner.words.len() as isize;
        let actual_idx = if idx < 0 { idx + len } else { idx };
        if actual_idx < 0 || actual_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        let u_idx = actual_idx as usize;
        Ok((
            self.inner.words[u_idx].clone(),
            self.inner.scores[u_idx],
        ))
    }

    pub fn __contains__(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    pub fn __add__(&self, other: &PyWordList) -> PyWordList {
        PyWordList {
            inner: Arc::new(self.inner.add(&other.inner)),
        }
    }
}

#[pyclass(name = "PyCrossword")]
#[derive(Clone)]
pub struct PyCrossword {
    pub core: CrosswordCore,
}

#[pymethods]
impl PyCrossword {
    #[new]
    #[pyo3(signature = (num_rows=None, num_cols=None, grid=None, symmetry=None, display_size_px=450))]
    pub fn new(
        num_rows: Option<usize>,
        num_cols: Option<usize>,
        grid: Option<Vec<Vec<String>>>,
        symmetry: Option<Symmetry>,
        display_size_px: u32,
    ) -> PyResult<Self> {
        if let Some(grid_data) = grid {
            let r = grid_data.len();
            if r == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("Grid cannot be empty"));
            }
            let c = grid_data[0].len();
            let mut cells = Vec::with_capacity(r * c);
            for row in grid_data {
                if row.len() != c {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "All rows in grid must have the same length",
                    ));
                }
                for val_str in row {
                    let cell_val = CellValue::parse(&val_str)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
                    cells.push(Cell::new(cell_val));
                }
            }
            Ok(PyCrossword {
                core: CrosswordCore::from_cells(r, c, cells, symmetry, display_size_px),
            })
        } else if let Some(r) = num_rows {
            let c = num_cols.unwrap_or(r);
            Ok(PyCrossword {
                core: CrosswordCore::new(r, c, symmetry, display_size_px),
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Either specify shape or provide grid.",
            ))
        }
    }

    #[getter]
    pub fn num_rows(&self) -> usize {
        self.core.grid.num_rows
    }

    #[getter]
    pub fn num_cols(&self) -> usize {
        self.core.grid.num_cols
    }

    #[getter]
    pub fn symmetry(&self) -> Option<Symmetry> {
        self.core.symmetry
    }

    #[setter]
    pub fn set_symmetry(&mut self, sym: Option<Symmetry>) {
        self.core.symmetry = sym;
    }

    #[getter]
    pub fn display_size_px(&self) -> u32 {
        self.core.display_size_px
    }

    #[setter]
    pub fn set_display_size_px(&mut self, px: u32) {
        self.core.display_size_px = px;
    }

    pub fn get_cell_value(&self, row: usize, col: usize) -> PyResult<String> {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("Cell index out of range"));
        }
        Ok(self.core.grid.get_cell(row, col).value.to_str())
    }

    pub fn get_cell_rebus(&self, row: usize, col: usize) -> Option<(String, String)> {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            return None;
        }
        match &self.core.grid.get_cell(row, col).value {
            CellValue::Rebus { across, down } => Some((across.clone(), down.clone())),
            _ => None,
        }
    }

    pub fn set_cell_rebus(&mut self, row: usize, col: usize, across: &str, down: &str) -> PyResult<()> {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("Cell index out of range"));
        }
        self.core.set_cell_rebus(row, col, across.to_uppercase(), down.to_uppercase());
        Ok(())
    }

    pub fn set_cell_value(&mut self, row: usize, col: usize, val_str: &str) -> PyResult<()> {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("Cell index out of range"));
        }
        let cell_val = CellValue::parse(val_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.core.set_cell(row, col, cell_val);
        Ok(())
    }

    pub fn get_cell_number(&self, row: usize, col: usize) -> Option<u16> {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            return None;
        }
        let num = self.core.grid.numbers[self.core.grid.idx(row, col)];
        if num > 0 {
            Some(num)
        } else {
            None
        }
    }

    pub fn is_cell_black(&self, row: usize, col: usize) -> bool {
        self.core.grid.is_black(row, col)
    }

    pub fn is_cell_open(&self, row: usize, col: usize) -> bool {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            false
        } else {
            self.core.grid.get_cell(row, col).is_open()
        }
    }

    pub fn get_cell_shaded(&self, row: usize, col: usize) -> bool {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            false
        } else {
            self.core.grid.get_cell(row, col).shaded
        }
    }

    pub fn set_cell_shaded(&mut self, row: usize, col: usize, shaded: bool) {
        if row < self.core.grid.num_rows && col < self.core.grid.num_cols {
            self.core.grid.get_cell_mut(row, col).shaded = shaded;
        }
    }

    pub fn get_cell_circled(&self, row: usize, col: usize) -> bool {
        if row >= self.core.grid.num_rows || col >= self.core.grid.num_cols {
            false
        } else {
            self.core.grid.get_cell(row, col).circled
        }
    }

    pub fn set_cell_circled(&mut self, row: usize, col: usize, circled: bool) {
        if row < self.core.grid.num_rows && col < self.core.grid.num_cols {
            self.core.grid.get_cell_mut(row, col).circled = circled;
        }
    }

    pub fn get_word_value(&self, direction: Direction, number: u16) -> Option<String> {
        self.core.grid.get_word_value((direction, number))
    }

    pub fn set_word_value(&mut self, direction: Direction, number: u16, value: &str) -> PyResult<()> {
        self.core
            .set_word((direction, number), value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    pub fn is_word_open(&self, direction: Direction, number: u16) -> bool {
        self.core.grid.is_word_open((direction, number))
    }

    pub fn get_word_length(&self, direction: Direction, number: u16) -> Option<usize> {
        self.core.grid.word_slots.get(&(direction, number)).map(|s| s.length)
    }

    pub fn get_word_cell_indices(&self, direction: Direction, number: u16) -> Option<Vec<(usize, usize)>> {
        self.core.grid.word_slots.get(&(direction, number)).map(|s| s.cell_indices.clone())
    }

    pub fn get_word_crosses(&self, direction: Direction, number: u16) -> Option<Vec<Option<(Direction, u16)>>> {
        let slot = self.core.grid.word_slots.get(&(direction, number))?;
        Some(slot.crosses.iter().map(|(w, _)| *w).collect())
    }

    pub fn get_word_at_cell(&self, row: usize, col: usize, direction: Direction) -> Option<(Direction, u16)> {
        self.core.grid.get_word_at_cell(row, col, direction)
    }

    pub fn get_clue(&self, direction: Direction, number: u16) -> String {
        self.core.clues.get(&(direction, number)).cloned().unwrap_or_default()
    }

    pub fn set_clue(&mut self, direction: Direction, number: u16, clue: String) {
        self.core.clues.insert((direction, number), clue);
    }

    pub fn get_clues(&self) -> Vec<((Direction, u16), String)> {
        let mut list: Vec<((Direction, u16), String)> = self
            .core
            .grid
            .word_slots
            .keys()
            .map(|&wi| (wi, self.get_clue(wi.0, wi.1)))
            .collect();
        list.sort_by_key(|(wi, _)| *wi);
        list
    }

    #[pyo3(signature = (direction=None, only_open=None))]
    pub fn iter_word_indices(&self, direction: Option<Direction>, only_open: Option<bool>) -> Vec<(Direction, u16)> {
        let open_flag = only_open.unwrap_or(false);
        let mut list: Vec<(Direction, u16)> = self
            .core
            .grid
            .word_slots
            .keys()
            .copied()
            .filter(|&wi| {
                if let Some(dir) = direction {
                    if wi.0 != dir {
                        return false;
                    }
                }
                if open_flag && !self.core.grid.is_word_open(wi) {
                    return false;
                }
                true
            })
            .collect();
        list.sort();
        list
    }

    pub fn get_symmetric_cell_indices(&self, row: usize, col: usize) -> Vec<(usize, usize)> {
        self.core.get_symmetric_cell_indices(row, col)
    }

    pub fn get_symmetric_word_indices(&self, direction: Direction, number: u16) -> Vec<(Direction, u16)> {
        self.core.get_symmetric_word_indices((direction, number))
    }

    pub fn get_disconnected_open_subgrids(&self) -> Vec<Vec<(Direction, u16)>> {
        self.core.grid.get_open_subgrids()
    }

    pub fn hashable_state(&self, word_indices: Vec<(Direction, u16)>) -> Vec<((Direction, u16), String)> {
        let mut res = Vec::with_capacity(word_indices.len());
        for wi in word_indices {
            let val = self.core.grid.get_word_value(wi).unwrap_or_default();
            res.push((wi, val));
        }
        res.sort_by_key(|(wi, _)| *wi);
        res
    }

    pub fn copy(&self) -> PyCrossword {
        PyCrossword {
            core: self.core.clone(),
        }
    }

    #[pyo3(signature = (word_list, timeout=None, temperature=None, score_filter=None, allow_repeats=None))]
    pub fn fill(
        &self,
        word_list: &PyWordList,
        timeout: Option<f64>,
        temperature: Option<f64>,
        score_filter: Option<f64>,
        allow_repeats: Option<bool>,
    ) -> Option<PyCrossword> {
        let temp = temperature.unwrap_or(0.0);
        let repeats = allow_repeats.unwrap_or(false);
        self.core
            .fill(&word_list.inner, timeout, temp, score_filter, repeats)
            .map(|core| PyCrossword { core })
    }

    pub fn grid_chars(&self) -> Vec<Vec<String>> {
        let rows = self.core.grid.num_rows;
        let cols = self.core.grid.num_cols;
        let mut res = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(self.core.grid.get_cell(r, c).value.to_str());
            }
            res.push(row);
        }
        res
    }

    pub fn numbers_grid(&self) -> Vec<Vec<u16>> {
        let rows = self.core.grid.num_rows;
        let cols = self.core.grid.num_cols;
        let mut res = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(self.core.grid.numbers[self.core.grid.idx(r, c)]);
            }
            res.push(row);
        }
        res
    }

    pub fn across_numbers_grid(&self) -> Vec<Vec<u16>> {
        let rows = self.core.grid.num_rows;
        let cols = self.core.grid.num_cols;
        let mut res = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(self.core.grid.across_numbers[self.core.grid.idx(r, c)]);
            }
            res.push(row);
        }
        res
    }

    pub fn down_numbers_grid(&self) -> Vec<Vec<u16>> {
        let rows = self.core.grid.num_rows;
        let cols = self.core.grid.num_cols;
        let mut res = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(self.core.grid.down_numbers[self.core.grid.idx(r, c)]);
            }
            res.push(row);
        }
        res
    }
}

#[pyfunction]
pub fn get_inverse_character_frequencies<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (i, &freq) in INVERSE_CHARACTER_FREQUENCIES.iter().enumerate() {
        let ch = ((b'A' + i as u8) as char).to_string();
        dict.set_item(ch, freq)?;
    }
    Ok(dict)
}

#[pymodule]
fn _blacksquare_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWordList>()?;
    m.add_class::<PyMatchWordList>()?;
    m.add_class::<PyCrossword>()?;
    m.add_class::<Direction>()?;
    m.add_class::<Symmetry>()?;
    m.add_function(wrap_pyfunction!(get_inverse_character_frequencies, m)?)?;
    Ok(())
}
