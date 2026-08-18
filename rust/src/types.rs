use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Across = 0,
    Down = 1,
}

#[pymethods]
impl Direction {
    #[getter]
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Across => Direction::Down,
            Direction::Down => Direction::Across,
        }
    }

    #[getter]
    pub fn value(&self) -> &'static str {
        match self {
            Direction::Across => "Across",
            Direction::Down => "Down",
        }
    }

    pub fn __repr__(&self) -> String {
        format!("<{}>", self.value())
    }

    pub fn __str__(&self) -> String {
        self.value().to_string()
    }
}

pub type WordIndex = (Direction, u16);
pub type CellIndex = (usize, usize);
