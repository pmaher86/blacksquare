import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from blacksquare.types import CellIndex


@dataclass
class SymmetryResult:
    """An object that contains a transformed grid, and an indicator for whether Across
    and Down have been flipped."""

    grid: np.ndarray
    word_direction_rotated: bool


class Symmetry(enum.Enum):
    """A class representing the possible symmetry types of a crossword grid."""

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

    def apply(
        self, grid: np.ndarray, force_list: bool = False
    ) -> Union[SymmetryResult, List[SymmetryResult]]:
        """Applies the symmetry group to an input array, and returns all images of the
        input under that symmetry.

        Args:
            grid (np.ndarray): The input grid.
            force_list (bool, optional): Whether to return single-image symmetry groups
                as lists, for consistent typing. Defaults to False.

        Returns:
            Union[SymmetryResult, List[SymmetryResult]]: If the symmetry type has only
            a single image and force_list is false, the return is a single
            SymmetryResult. Otherwise, the result is a list of all SymmetryResults.
        """
        if self == Symmetry.ROTATIONAL:
            images = SymmetryResult(np.rot90(grid, k=2), False)
        elif self == Symmetry.FULL:
            images = [
                SymmetryResult(np.fliplr(grid), False),
                SymmetryResult(np.flipud(grid), False),
                SymmetryResult(np.fliplr(np.flipud(grid)), False),
                SymmetryResult(np.transpose(grid), True),
                SymmetryResult(np.transpose(np.fliplr(grid)), True),
                SymmetryResult(np.transpose(np.flipud(grid)), True),
                SymmetryResult(np.transpose(np.fliplr(np.flipud(grid))), True),
            ]
        elif self == Symmetry.VERTICAL:
            images = SymmetryResult(np.fliplr(grid), False)
        elif self == Symmetry.HORIZONTAL:
            images = SymmetryResult(np.flipud(grid), False)
        elif self == Symmetry.BIAXIAL:
            images = [
                SymmetryResult(np.fliplr(grid), False),
                SymmetryResult(np.flipud(grid), False),
                SymmetryResult(np.fliplr(np.flipud(grid)), False),
            ]
        elif self == Symmetry.NE_DIAGONAL:
            images = SymmetryResult(np.transpose(np.rot90(grid, k=2)), True)
        elif self == Symmetry.NW_DIAGONAL:
            images = SymmetryResult(np.transpose(grid), True)

        if not isinstance(images, list) and force_list:
            return [images]
        else:
            return images
