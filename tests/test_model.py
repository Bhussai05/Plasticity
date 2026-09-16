import pytest

from model import pop_rate

def test_zero_population():
    assert pop_rate(0.0, 1.0) == pytest.approx(0.0)

def test_below_capacity():
    assert pop_rate(0.1, 1.0) == pytest.approx(0.09)

def test_at_capacity():
    assert pop_rate(1.0, 1.0) == pytest.approx(0.0)

def test_above_capacity():
    assert pop_rate(1.5, 1.0) == pytest.approx(-0.75)

def test_different_capacity():
    assert pop_rate(2.0, 4.0) == pytest.approx(1.0)