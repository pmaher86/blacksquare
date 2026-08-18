use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Symmetry {
    Rotational = 0,
    Full = 1,
    Vertical = 2,
    Horizontal = 3,
    Biaxial = 4,
    NeDiagonal = 5,
    NwDiagonal = 6,
}

#[pymethods]
impl Symmetry {
    #[getter]
    pub fn value(&self) -> &'static str {
        match self {
            Symmetry::Rotational => "rotational",
            Symmetry::Full => "full",
            Symmetry::Vertical => "vertical",
            Symmetry::Horizontal => "horizontal",
            Symmetry::Biaxial => "biaxial",
            Symmetry::NeDiagonal => "ne_diagonal",
            Symmetry::NwDiagonal => "nw_diagonal",
        }
    }

    #[getter]
    pub fn is_multi_image(&self) -> bool {
        matches!(self, Symmetry::Full | Symmetry::Biaxial)
    }

    #[getter]
    pub fn requires_square(&self) -> bool {
        matches!(self, Symmetry::Full | Symmetry::NeDiagonal | Symmetry::NwDiagonal)
    }
}

pub struct SymmetryImage {
    pub cell_index: (usize, usize),
    pub word_direction_rotated: bool,
}

impl Symmetry {
    pub fn apply_cell(&self, row: usize, col: usize, num_rows: usize, num_cols: usize) -> Vec<SymmetryImage> {
        let max_r = num_rows - 1;
        let max_c = num_cols - 1;

        match self {
            Symmetry::Rotational => vec![SymmetryImage {
                cell_index: (max_r - row, max_c - col),
                word_direction_rotated: false,
            }],
            Symmetry::Vertical => vec![SymmetryImage {
                cell_index: (row, max_c - col),
                word_direction_rotated: false,
            }],
            Symmetry::Horizontal => vec![SymmetryImage {
                cell_index: (max_r - row, col),
                word_direction_rotated: false,
            }],
            Symmetry::Biaxial => vec![
                SymmetryImage {
                    cell_index: (row, max_c - col),
                    word_direction_rotated: false,
                },
                SymmetryImage {
                    cell_index: (max_r - row, col),
                    word_direction_rotated: false,
                },
                SymmetryImage {
                    cell_index: (max_r - row, max_c - col),
                    word_direction_rotated: false,
                },
            ],
            Symmetry::NwDiagonal => vec![SymmetryImage {
                cell_index: (col, row),
                word_direction_rotated: true,
            }],
            Symmetry::NeDiagonal => vec![SymmetryImage {
                cell_index: (max_c - col, max_r - row),
                word_direction_rotated: true,
            }],
            Symmetry::Full => vec![
                SymmetryImage {
                    cell_index: (row, max_c - col),
                    word_direction_rotated: false,
                },
                SymmetryImage {
                    cell_index: (max_r - row, col),
                    word_direction_rotated: false,
                },
                SymmetryImage {
                    cell_index: (max_r - row, max_c - col),
                    word_direction_rotated: false,
                },
                SymmetryImage {
                    cell_index: (col, row),
                    word_direction_rotated: true,
                },
                SymmetryImage {
                    cell_index: (col, max_r - row),
                    word_direction_rotated: true,
                },
                SymmetryImage {
                    cell_index: (max_c - col, row),
                    word_direction_rotated: true,
                },
                SymmetryImage {
                    cell_index: (max_c - col, max_r - row),
                    word_direction_rotated: true,
                },
            ],
        }
    }
}
