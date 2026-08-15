use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod word_list;
use word_list::{FastWordList, MatchWordListCore, INVERSE_CHARACTER_FREQUENCIES};

#[pyclass(name = "PyMatchWordList")]
#[derive(Clone)]
pub struct PyMatchWordList {
    inner: MatchWordListCore,
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
    inner: FastWordList,
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
                inner: FastWordList::new(entries),
            });
        }

        if let Ok(list) = source.downcast::<PyList>() {
            let mut entries = Vec::with_capacity(list.len());
            for item in list.iter() {
                let word: String = item.extract()?;
                entries.push((word, 1.0));
            }
            return Ok(PyWordList {
                inner: FastWordList::new(entries),
            });
        }

        if let Ok(path_str) = source.extract::<String>() {
            let inner = FastWordList::from_dict_file(&path_str)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            return Ok(PyWordList { inner });
        }

        Err(pyo3::exceptions::PyValueError::new_err(
            "Input type not recognized for PyWordList",
        ))
    }

    #[staticmethod]
    pub fn from_words_scores(words: Vec<String>, scores: Vec<f64>) -> Self {
        let entries: Vec<(String, f64)> = words.into_iter().zip(scores.into_iter()).collect();
        PyWordList {
            inner: FastWordList::new(entries),
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
            inner: self.inner.score_filter(threshold),
        }
    }

    pub fn get_partition(&self, length: usize) -> Option<(Vec<String>, Vec<f64>)> {
        self.inner.get_partition(length)
    }

    /// Fast native cross-scoring for a crossword word.
    /// Evaluates candidate words against per-letter cross distribution weights in Rust.
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
            inner: self.inner.add(&other.inner),
        }
    }
}

/// Helper function exposed to Python for inverse character frequencies.
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
    m.add_function(wrap_pyfunction!(get_inverse_character_frequencies, m)?)?;
    Ok(())
}
