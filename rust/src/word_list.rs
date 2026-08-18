use ahash::AHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

/// Character inverse frequencies for cross-matching ranking.
pub const INVERSE_CHARACTER_FREQUENCIES: [f64; 26] = [
    0.45,  // A
    1.98,  // B
    0.99,  // C
    1.09,  // D
    0.34,  // E
    2.27,  // F
    1.43,  // G
    1.26,  // H
    0.50,  // I
    14.81, // J
    3.21,  // K
    0.75,  // L
    1.34,  // M
    0.57,  // N
    0.53,  // O
    1.41,  // P
    27.79, // Q
    0.56,  // R
    0.51,  // S
    0.55,  // T
    1.19,  // U
    3.76,  // V
    3.18,  // W
    10.08, // X
    2.21,  // Y
    12.62, // Z
];

/// A partition of words of a specific fixed length.
#[derive(Clone)]
pub struct LengthPartition {
    pub length: usize,
    pub count: usize,
    /// Vector of words in the partition.
    pub words: Vec<String>,
    /// Flat contiguous ASCII buffer: `length * count` bytes.
    pub words_bytes: Vec<u8>,
    /// Normalized scores: `count` values.
    pub scores: Vec<f64>,
    /// Inverted bitmasks: `length * 26` bitvectors.
    /// Each bitvector has `mask_words_len` u64 elements.
    pub masks: Vec<Vec<u64>>,
    pub mask_words_len: usize,
    /// Fast word to index lookup within this partition.
    pub word_indices: AHashMap<String, usize>,
}

impl LengthPartition {
    pub fn new(length: usize, words: &[String], scores: &[f64]) -> Self {
        let count = words.len();
        let mask_words_len = (count + 63) / 64;
        let mut words_bytes = Vec::with_capacity(length * count);
        let mut part_scores = Vec::with_capacity(count);
        let mut word_indices = AHashMap::with_capacity(count);

        let mut masks = vec![vec![0u64; mask_words_len]; length * 26];

        for (idx, (w, &s)) in words.iter().zip(scores.iter()).enumerate() {
            let bytes = w.as_bytes();
            words_bytes.extend_from_slice(bytes);
            part_scores.push(s);
            word_indices.insert(w.clone(), idx);

            let word_chunk = idx / 64;
            let bit_pos = idx % 64;
            let bit = 1u64 << bit_pos;

            for pos in 0..length {
                let byte = bytes[pos];
                if byte >= b'A' && byte <= b'Z' {
                    let char_idx = (byte - b'A') as usize;
                    masks[pos * 26 + char_idx][word_chunk] |= bit;
                }
            }
        }

        LengthPartition {
            length,
            count,
            words: words.to_vec(),
            words_bytes,
            scores: part_scores,
            masks,
            mask_words_len,
            word_indices,
        }
    }

    /// Match a pattern with wildcards ('?', ' ', '_', '-').
    /// Computes the result bitmask.
    pub fn match_pattern(&self, query: &[u8], out_mask: &mut Vec<u64>) {
        let constraints: Vec<(usize, usize)> = query
            .iter()
            .enumerate()
            .filter_map(|(pos, &b)| {
                if b >= b'A' && b <= b'Z' {
                    Some((pos, (b - b'A') as usize))
                } else {
                    None
                }
            })
            .collect();

        if constraints.is_empty() {
            out_mask.resize(self.mask_words_len, 0);
            out_mask.fill(!0u64);
            let remainder = self.count % 64;
            if remainder != 0 {
                let last = self.mask_words_len - 1;
                out_mask[last] = (1u64 << remainder) - 1;
            }
            return;
        }

        let first = constraints[0];
        let first_mask = &self.masks[first.0 * 26 + first.1];
        out_mask.clear();
        out_mask.extend_from_slice(first_mask);

        for &(pos, char_idx) in &constraints[1..] {
            let mask = &self.masks[pos * 26 + char_idx];
            for (dst, src) in out_mask.iter_mut().zip(mask.iter()) {
                *dst &= *src;
            }
        }
    }

    /// Fast early-exit check to determine if any word matches the pattern.
    #[inline(always)]
    pub fn has_match(&self, query: &[u8]) -> bool {
        let mut constraints = [0usize; 32];
        let mut n_constraints = 0;

        for (pos, &b) in query.iter().enumerate() {
            if b >= b'A' && b <= b'Z' {
                let char_idx = (b - b'A') as usize;
                constraints[n_constraints] = pos * 26 + char_idx;
                n_constraints += 1;
            }
        }

        if n_constraints == 0 {
            return self.count > 0;
        }

        let first_mask = &self.masks[constraints[0]];
        if n_constraints == 1 {
            return first_mask.iter().any(|&w| w != 0);
        }

        for i in 0..self.mask_words_len {
            let mut w = first_mask[i];
            if w == 0 {
                continue;
            }
            for &mask_idx in &constraints[1..n_constraints] {
                w &= self.masks[mask_idx][i];
                if w == 0 {
                    break;
                }
            }
            if w != 0 {
                return true;
            }
        }
        false
    }

    /// Extract matching words and scores from a bitmask.
    pub fn extract_matches(&self, mask: &[u64]) -> (Vec<String>, Vec<f64>) {
        let count_matches: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        let mut words = Vec::with_capacity(count_matches);
        let mut scores = Vec::with_capacity(count_matches);
        let stride = self.length;

        for (chunk_idx, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let bit_pos = b.trailing_zeros() as usize;
                let word_idx = chunk_idx * 64 + bit_pos;
                if word_idx >= self.count {
                    break;
                }
                let offset = word_idx * stride;
                let word_slice = &self.words_bytes[offset..offset + stride];
                let word_str = unsafe { std::str::from_utf8_unchecked(word_slice) };
                words.push(word_str.to_string());
                scores.push(self.scores[word_idx]);
                b &= b - 1;
            }
        }

        (words, scores)
    }

    /// Sum scores grouped by letter at a given column index.
    pub fn letter_scores_at_index(&self, mask: &[u64], index: usize) -> [f64; 26] {
        let mut letter_scores = [0.0f64; 26];
        if index >= self.length {
            return letter_scores;
        }

        let stride = self.length;
        for (chunk_idx, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let bit_pos = b.trailing_zeros() as usize;
                let word_idx = chunk_idx * 64 + bit_pos;
                if word_idx >= self.count {
                    break;
                }
                let ch_byte = self.words_bytes[word_idx * stride + index];
                if ch_byte >= b'A' && ch_byte <= b'Z' {
                    let ch_idx = (ch_byte - b'A') as usize;
                    letter_scores[ch_idx] += self.scores[word_idx];
                }
                b &= b - 1;
            }
        }

        letter_scores
    }

    #[inline(always)]
    pub fn get_word_bytes(&self, word_idx: usize) -> &[u8] {
        let offset = word_idx * self.length;
        &self.words_bytes[offset..offset + self.length]
    }

    #[inline(always)]
    pub fn get_word_str(&self, word_idx: usize) -> &str {
        let slice = self.get_word_bytes(word_idx);
        unsafe { std::str::from_utf8_unchecked(slice) }
    }

    /// Fused cross scoring returning sorted word indices (zero String allocation).
    pub fn fused_cross_score_indices(
        &self,
        mask: &[u64],
        open_positions: &[usize],
        letter_weights_per_pos: &[[f64; 26]],
        drop_zeros: bool,
    ) -> Vec<usize> {
        let count_matches: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        let mut scored_indices: Vec<(usize, f64)> = Vec::with_capacity(count_matches);
        let stride = self.length;

        for (chunk_idx, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let bit_pos = b.trailing_zeros() as usize;
                let word_idx = chunk_idx * 64 + bit_pos;
                if word_idx >= self.count {
                    break;
                }
                let word_score = self.scores[word_idx];
                let word_offset = word_idx * stride;

                let mut prod = 1.0f64;
                for (open_idx, &pos) in open_positions.iter().enumerate() {
                    let ch_byte = self.words_bytes[word_offset + pos];
                    if ch_byte >= b'A' && ch_byte <= b'Z' {
                        let ch = (ch_byte - b'A') as usize;
                        prod *= letter_weights_per_pos[open_idx][ch];
                    }
                }

                let final_score = (prod + 1.0).ln() * word_score;
                if !drop_zeros || final_score > 0.0 {
                    scored_indices.push((word_idx, final_score));
                }
                b &= b - 1;
            }
        }

        scored_indices.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        scored_indices.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Fused cross scoring across open positions.
    pub fn fused_cross_score(
        &self,
        mask: &[u64],
        open_positions: &[usize],
        letter_weights_per_pos: &[[f64; 26]],
        drop_zeros: bool,
    ) -> (Vec<String>, Vec<f64>) {
        let count_matches: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        let mut scored_indices: Vec<(usize, f64)> = Vec::with_capacity(count_matches);
        let stride = self.length;

        for (chunk_idx, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let bit_pos = b.trailing_zeros() as usize;
                let word_idx = chunk_idx * 64 + bit_pos;
                if word_idx >= self.count {
                    break;
                }
                let word_score = self.scores[word_idx];
                let word_offset = word_idx * stride;

                let mut prod = 1.0f64;
                for (open_idx, &pos) in open_positions.iter().enumerate() {
                    let ch_byte = self.words_bytes[word_offset + pos];
                    if ch_byte >= b'A' && ch_byte <= b'Z' {
                        let ch = (ch_byte - b'A') as usize;
                        prod *= letter_weights_per_pos[open_idx][ch];
                    }
                }

                let final_score = (prod + 1.0).ln() * word_score;
                if !drop_zeros || final_score > 0.0 {
                    scored_indices.push((word_idx, final_score));
                }
                b &= b - 1;
            }
        }

        scored_indices.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut words = Vec::with_capacity(scored_indices.len());
        let mut scores = Vec::with_capacity(scored_indices.len());

        for (word_idx, score) in scored_indices {
            let offset = word_idx * stride;
            let word_slice = &self.words_bytes[offset..offset + stride];
            let word_str = unsafe { std::str::from_utf8_unchecked(word_slice) };
            words.push(word_str.to_string());
            scores.push(score);
        }

        (words, scores)
    }
}

/// The core WordList data structure.
#[derive(Clone)]
pub struct FastWordList {
    pub words: Vec<String>,
    pub scores: Vec<f64>,
    pub partitions: Vec<Option<Arc<LengthPartition>>>,
    pub word_map: AHashMap<String, f64>,
}

impl FastWordList {
    pub fn new(raw_entries: Vec<(String, f64)>) -> Self {
        let mut filtered: Vec<(String, f64)> = Vec::with_capacity(raw_entries.len());
        for (w, s) in raw_entries {
            let norm = normalize_word(&w);
            if is_alpha(&norm) && !norm.is_empty() {
                filtered.push((norm, s));
            }
        }

        if filtered.is_empty() {
            return FastWordList {
                words: Vec::new(),
                scores: Vec::new(),
                partitions: vec![None; 33],
                word_map: AHashMap::new(),
            };
        }

        filtered.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let max_score = filtered
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0f64, |acc, x| acc.max(x));
        let scale = if max_score > 0.0 { 1.0 / max_score } else { 1.0 };

        let mut words = Vec::with_capacity(filtered.len());
        let mut scores = Vec::with_capacity(filtered.len());
        let mut word_map = AHashMap::with_capacity(filtered.len());

        let mut words_by_length: Vec<Vec<String>> = vec![Vec::new(); 33];
        let mut scores_by_length: Vec<Vec<f64>> = vec![Vec::new(); 33];

        for (w, s) in filtered {
            let norm_score = s * scale;
            let len = w.len();
            if len < 33 {
                words_by_length[len].push(w.clone());
                scores_by_length[len].push(norm_score);
            }
            word_map.insert(w.clone(), norm_score);
            words.push(w);
            scores.push(norm_score);
        }

        let mut partitions = vec![None; 33];
        for len in 1..33 {
            if !words_by_length[len].is_empty() {
                partitions[len] = Some(Arc::new(LengthPartition::new(
                    len,
                    &words_by_length[len],
                    &scores_by_length[len],
                )));
            }
        }

        FastWordList {
            words,
            scores,
            partitions,
            word_map,
        }
    }

    pub fn from_dict_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((w, s)) = trimmed.split_once(';') {
                if let Ok(score) = s.trim().parse::<f64>() {
                    entries.push((w.trim().to_string(), score));
                }
            } else {
                entries.push((trimmed.to_string(), 1.0));
            }
        }

        Ok(Self::new(entries))
    }

    pub fn default_embedded() -> Arc<FastWordList> {
        static DEFAULT_WORDLIST_STATIC: std::sync::OnceLock<Arc<FastWordList>> =
            std::sync::OnceLock::new();
        DEFAULT_WORDLIST_STATIC
            .get_or_init(|| {
                static BYTES: &[u8] = include_bytes!("../data/spreadthewordlist.bin.gz");
                Arc::new(FastWordList::from_compressed_binary(BYTES))
            })
            .clone()
    }

    pub fn from_compressed_binary(gz_bytes: &[u8]) -> Self {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(gz_bytes);
        let mut uncompressed = Vec::new();
        decoder
            .read_to_end(&mut uncompressed)
            .expect("Failed to decompress embedded wordlist");

        let mut offset = 0;
        let count =
            u32::from_le_bytes(uncompressed[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let mut entries: Vec<(String, f64)> = Vec::with_capacity(count);
        for _ in 0..count {
            let len = uncompressed[offset] as usize;
            offset += 1;
            let word_bytes = &uncompressed[offset..offset + len];
            offset += len;
            let score_bytes: [u8; 4] = uncompressed[offset..offset + 4].try_into().unwrap();
            offset += 4;
            let score = f32::from_le_bytes(score_bytes) as f64;
            let word = unsafe { String::from_utf8_unchecked(word_bytes.to_vec()) };
            entries.push((word, score));
        }

        FastWordList::new(entries)
    }

    pub fn find_matches_str(&self, query: &str) -> MatchWordListCore {
        let clean_query = query.to_uppercase();
        let query_bytes: Vec<u8> = clean_query
            .bytes()
            .map(|b| match b {
                b'?' | b' ' | b'_' | b'-' => b'?',
                _ => b,
            })
            .collect();
        let len = query_bytes.len();

        if len == 0 || len >= self.partitions.len() {
            return MatchWordListCore::empty(len);
        }

        if let Some(part) = &self.partitions[len] {
            let mut mask = Vec::new();
            part.match_pattern(&query_bytes, &mut mask);
            MatchWordListCore::from_bitmask(Arc::clone(part), mask)
        } else {
            MatchWordListCore::empty(len)
        }
    }

    pub fn get_score(&self, word: &str) -> Option<f64> {
        let norm = normalize_word(word);
        self.word_map.get(&norm).copied()
    }

    pub fn contains(&self, word: &str) -> bool {
        let norm = normalize_word(word);
        self.word_map.contains_key(&norm)
    }

    pub fn score_filter(&self, threshold: f64) -> Self {
        let mut partitions = vec![None; 33];
        let mut words = Vec::new();
        let mut scores = Vec::new();
        let mut word_map = AHashMap::new();

        for len in 1..33 {
            if let Some(part) = &self.partitions[len] {
                let cutoff = part.scores.partition_point(|&s| s >= threshold);
                if cutoff > 0 {
                    let sub_words = &part.words[..cutoff];
                    let sub_scores = &part.scores[..cutoff];
                    for (w, &s) in sub_words.iter().zip(sub_scores.iter()) {
                        words.push(w.clone());
                        scores.push(s);
                        word_map.insert(w.clone(), s);
                    }
                    partitions[len] = Some(Arc::new(LengthPartition::new(len, sub_words, sub_scores)));
                }
            }
        }

        FastWordList {
            words,
            scores,
            partitions,
            word_map,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut map: AHashMap<String, f64> = self.word_map.clone();
        for (w, &s) in &other.word_map {
            map.insert(w.clone(), s);
        }
        let entries: Vec<(String, f64)> = map.into_iter().collect();
        Self::new(entries)
    }

    pub fn get_partition(&self, length: usize) -> Option<(Vec<String>, Vec<f64>)> {
        if length >= self.partitions.len() {
            return None;
        }
        self.partitions[length].as_ref().map(|p| (p.words.clone(), p.scores.clone()))
    }
}

/// Representation of matches from a search query.
#[derive(Clone)]
pub enum MatchState {
    Bitmask {
        partition: Arc<LengthPartition>,
        mask: Vec<u64>,
        match_count: usize,
    },
    Materialized {
        words: Vec<String>,
        scores: Vec<f64>,
        word_map: AHashMap<String, f64>,
    },
}

#[derive(Clone)]
pub struct MatchWordListCore {
    pub word_length: usize,
    pub state: MatchState,
}

impl MatchWordListCore {
    pub fn from_bitmask(partition: Arc<LengthPartition>, mask: Vec<u64>) -> Self {
        let length = partition.length;
        let match_count = mask.iter().map(|w| w.count_ones() as usize).sum();
        MatchWordListCore {
            word_length: length,
            state: MatchState::Bitmask {
                partition,
                mask,
                match_count,
            },
        }
    }

    pub fn materialized(word_length: usize, words: Vec<String>, scores: Vec<f64>) -> Self {
        let mut word_map = AHashMap::with_capacity(words.len());
        for (w, &s) in words.iter().zip(scores.iter()) {
            word_map.insert(w.clone(), s);
        }
        MatchWordListCore {
            word_length,
            state: MatchState::Materialized {
                words,
                scores,
                word_map,
            },
        }
    }

    pub fn empty(word_length: usize) -> Self {
        MatchWordListCore::materialized(word_length, Vec::new(), Vec::new())
    }

    pub fn len(&self) -> usize {
        match &self.state {
            MatchState::Bitmask { match_count, .. } => *match_count,
            MatchState::Materialized { words, .. } => words.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn extract_words_and_scores(&self) -> (Vec<String>, Vec<f64>) {
        match &self.state {
            MatchState::Bitmask {
                partition, mask, ..
            } => partition.extract_matches(mask),
            MatchState::Materialized { words, scores, .. } => (words.clone(), scores.clone()),
        }
    }

    pub fn words(&self) -> Vec<String> {
        self.extract_words_and_scores().0
    }

    pub fn scores(&self) -> Vec<f64> {
        self.extract_words_and_scores().1
    }

    pub fn letter_scores_at_index(&self, index: usize) -> [f64; 26] {
        let mut letter_scores = [0.0f64; 26];
        if index >= self.word_length {
            return letter_scores;
        }

        match &self.state {
            MatchState::Bitmask {
                partition, mask, ..
            } => partition.letter_scores_at_index(mask, index),
            MatchState::Materialized { words, scores, .. } => {
                for (w, &s) in words.iter().zip(scores.iter()) {
                    let bytes = w.as_bytes();
                    if index < bytes.len() {
                        let ch = bytes[index];
                        if ch >= b'A' && ch <= b'Z' {
                            letter_scores[(ch - b'A') as usize] += s;
                        }
                    }
                }
                letter_scores
            }
        }
    }

    pub fn filter_words(&self, words_to_exclude: &[String]) -> Self {
        match &self.state {
            MatchState::Bitmask {
                partition, mask, ..
            } => {
                let mut new_mask = mask.clone();
                for w in words_to_exclude {
                    if let Some(&idx) = partition.word_indices.get(w) {
                        let chunk = idx / 64;
                        let bit = idx % 64;
                        if chunk < new_mask.len() {
                            new_mask[chunk] &= !(1u64 << bit);
                        }
                    }
                }
                MatchWordListCore::from_bitmask(Arc::clone(partition), new_mask)
            }
            MatchState::Materialized { words, scores, .. } => {
                let exclude_set: ahash::AHashSet<&str> =
                    words_to_exclude.iter().map(|s| s.as_str()).collect();

                let mut out_words = Vec::new();
                let mut out_scores = Vec::new();

                for (w, &s) in words.iter().zip(scores.iter()) {
                    if !exclude_set.contains(w.as_str()) {
                        out_words.push(w.clone());
                        out_scores.push(s);
                    }
                }
                MatchWordListCore::materialized(self.word_length, out_words, out_scores)
            }
        }
    }

    pub fn score_filter(&self, threshold: f64) -> Self {
        let (words, scores) = self.extract_words_and_scores();
        let mut out_words = Vec::new();
        let mut out_scores = Vec::new();

        for (w, s) in words.into_iter().zip(scores.into_iter()) {
            if s >= threshold {
                out_words.push(w);
                out_scores.push(s);
            }
        }

        MatchWordListCore::materialized(self.word_length, out_words, out_scores)
    }

    pub fn get_score(&self, word: &str) -> Option<f64> {
        let norm = normalize_word(word);
        match &self.state {
            MatchState::Bitmask {
                partition, mask, ..
            } => {
                if let Some(&idx) = partition.word_indices.get(&norm) {
                    let chunk = idx / 64;
                    let bit = idx % 64;
                    if chunk < mask.len() && (mask[chunk] & (1u64 << bit)) != 0 {
                        return Some(partition.scores[idx]);
                    }
                }
                None
            }
            MatchState::Materialized { word_map, .. } => word_map.get(&norm).copied(),
        }
    }

    pub fn get_item(&self, index: usize) -> Option<(String, f64)> {
        match &self.state {
            MatchState::Bitmask {
                partition, mask, ..
            } => {
                let mut current_idx = 0;
                let stride = partition.length;
                for (chunk_idx, &bits) in mask.iter().enumerate() {
                    let mut b = bits;
                    while b != 0 {
                        let bit_pos = b.trailing_zeros() as usize;
                        let word_idx = chunk_idx * 64 + bit_pos;
                        if word_idx >= partition.count {
                            break;
                        }
                        if current_idx == index {
                            let offset = word_idx * stride;
                            let word_slice = &partition.words_bytes[offset..offset + stride];
                            let word_str = unsafe { std::str::from_utf8_unchecked(word_slice) };
                            return Some((word_str.to_string(), partition.scores[word_idx]));
                        }
                        current_idx += 1;
                        b &= b - 1;
                    }
                }
                None
            }
            MatchState::Materialized { words, scores, .. } => {
                if index < words.len() {
                    Some((words[index].clone(), scores[index]))
                } else {
                    None
                }
            }
        }
    }
}

pub fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

pub fn is_alpha(word: &str) -> bool {
    word.chars().all(|c| c.is_ascii_uppercase())
}
