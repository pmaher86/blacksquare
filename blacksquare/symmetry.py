import enum
from blacksquare.types import CellIndex
from typing import Tuple, Optional, Union, List
import itertools
import numpy as np
from dataclasses import dataclass


@dataclass
class SymmetryResult:
    grid: np.ndarray
    word_direction_rotated: bool


class Symmetry(enum.Enum):
    ROTATIONAL = "rotational"
    FULL = "full"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    BIAXIAL = "biaxial"
    NE_DIAGONAL = "ne_diagonal"
    NW_DIAGONAL = "nw_diagonal"

    @property
    def is_multi_image(self) -> bool:
        return self in {Symmetry.FULL, Symmetry.BIAXIAL}

    @property
    def requires_square(self) -> bool:
        return self in {Symmetry.FULL, Symmetry.NE_DIAGONAL, Symmetry.NW_DIAGONAL}

    def apply(self, grid: np.ndarray) -> Union[SymmetryResult, List[SymmetryResult]]:
        if self == Symmetry.ROTATIONAL:
            return SymmetryResult(np.rot90(grid, k=2), False)
        elif self == Symmetry.FULL:
            return [
                SymmetryResult(np.fliplr(grid), False),
                SymmetryResult(np.flipud(grid), False),
                SymmetryResult(np.fliplr(np.flipud(grid)), False),
                SymmetryResult(np.transpose(grid), True),
                SymmetryResult(np.transpose(np.fliplr(grid)), True),
                SymmetryResult(np.transpose(np.flipud(grid)), True),
                SymmetryResult(np.transpose(np.fliplr(np.flipud(grid))), True),
            ]
        elif self == Symmetry.VERTICAL:
            return SymmetryResult(np.fliplr(grid), False)
        elif self == Symmetry.HORIZONTAL:
            return SymmetryResult(np.flipud(grid), False)
        elif self == Symmetry.BIAXIAL:
            return [
                SymmetryResult(np.fliplr(grid), False),
                SymmetryResult(np.flipud(grid), False),
                SymmetryResult(np.fliplr(np.flipud(grid)), False),
            ]
        elif self == Symmetry.NE_DIAGONAL:
            return SymmetryResult(np.transpose(np.rot90(grid, k=2)), True)
        elif self == Symmetry.NW_DIAGONAL:
            return SymmetryResult(np.transpose(grid), True)
