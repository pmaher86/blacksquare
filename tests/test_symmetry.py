import numpy as np
import pytest

from blacksquare.symmetry import Symmetry


@pytest.fixture
def rect():
    return np.arange(35).reshape(5, 7)


@pytest.fixture
def square():
    return np.arange(49).reshape(7, 7)


def test_rotation(rect):
    image = Symmetry.ROTATIONAL.apply(rect).grid
    assert rect[0, 1] == image[4, 5]
    assert rect[2, 3] == image[2, 3]


def test_full(square):
    images = [x.grid for x in Symmetry.FULL.apply(square)]
    assert set(i[0, 1] for i in images) == {
        square[1, 0],
        square[5, 6],
        square[6, 5],
        square[0, 5],
        square[5, 0],
        square[6, 1],
        square[1, 6],
    }
    assert set(i[0, 3] for i in images) == {
        square[0, 3],
        square[3, 0],
        square[3, 6],
        square[6, 3],
    }
    assert set(i[3, 3] for i in images) == {square[3, 3]}


def test_vertical(rect):
    image = Symmetry.VERTICAL.apply(rect).grid
    assert rect[0, 1] == image[0, 5]
    assert np.all(rect[:, 3] == image[:, 3])


def test_horizontal(rect):
    image = Symmetry.HORIZONTAL.apply(rect).grid
    assert rect[0, 1] == image[4, 1]
    assert np.all(rect[2, :] == image[2, :])


def test_biaxial(rect):
    images = [x.grid for x in Symmetry.BIAXIAL.apply(rect)]
    assert set(i[0, 1] for i in images) == {rect[4, 1], rect[0, 5], rect[4, 5]}
    assert set(i[0, 3] for i in images) == {rect[0, 3], rect[4, 3]}
    assert set(i[2, 3] for i in images) == {rect[2, 3]}


def test_ne_diagonal(square):
    image = Symmetry.NE_DIAGONAL.apply(square).grid
    assert square[0, 1] == image[5, 6]
    assert square[0, 6] == image[0, 6]


def test_nw_diagonal(square):
    image = Symmetry.NW_DIAGONAL.apply(square).grid
    assert square[0, 1] == image[1, 0]
    assert square[2, 2] == image[2, 2]
