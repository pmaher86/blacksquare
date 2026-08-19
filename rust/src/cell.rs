use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CellValue {
    Empty,
    Black,
    Letter(char),
    Rebus { across: String, down: String },
    Schrodinger(Vec<String>),
}

impl CellValue {
    pub fn parse(input: &str) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == " " || trimmed == "?" || trimmed == "_" || trimmed == "-" {
            return Ok(CellValue::Empty);
        }
        if trimmed == "." || trimmed == "#" || trimmed == "█" {
            return Ok(CellValue::Black);
        }

        let upper = trimmed.to_uppercase();
        let chars: Vec<char> = upper.chars().collect();
        if chars.len() == 1 {
            Ok(CellValue::Letter(chars[0]))
        } else if upper.contains('/') {
            let parts: Vec<String> = upper
                .split('/')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() == 2 {
                Ok(CellValue::Rebus {
                    across: parts[0].clone(),
                    down: parts[1].clone(),
                })
            } else if parts.len() > 2 {
                Ok(CellValue::Schrodinger(parts))
            } else {
                Ok(CellValue::Rebus {
                    across: upper.clone(),
                    down: upper,
                })
            }
        } else {
            Ok(CellValue::Rebus {
                across: upper.clone(),
                down: upper,
            })
        }
    }

    pub fn to_str(&self) -> String {
        match self {
            CellValue::Empty => " ".to_string(),
            CellValue::Black => "█".to_string(),
            CellValue::Letter(c) => c.to_string(),
            CellValue::Rebus { across, down } => {
                if across == down {
                    across.clone()
                } else {
                    format!("{}/{}", across, down)
                }
            }
            CellValue::Schrodinger(parts) => parts.join("/"),
        }
    }

    pub fn char_or_wildcard(&self) -> char {
        match self {
            CellValue::Empty => '?',
            CellValue::Black => '#',
            CellValue::Letter(c) => *c,
            CellValue::Rebus { across, .. } => across.chars().next().unwrap_or('?'),
            CellValue::Schrodinger(parts) => {
                parts.first().and_then(|s| s.chars().next()).unwrap_or('?')
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, CellValue::Empty)
    }

    pub fn is_black(&self) -> bool {
        matches!(self, CellValue::Black)
    }
}

impl fmt::Display for CellValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub value: CellValue,
    pub shaded: bool,
    pub circled: bool,
}

impl Cell {
    pub fn new(value: CellValue) -> Self {
        Cell {
            value,
            shaded: false,
            circled: false,
        }
    }

    pub fn empty() -> Self {
        Self::new(CellValue::Empty)
    }

    pub fn black() -> Self {
        Self::new(CellValue::Black)
    }

    pub fn is_open(&self) -> bool {
        self.value.is_empty()
    }

    pub fn is_black(&self) -> bool {
        self.value.is_black()
    }
}
