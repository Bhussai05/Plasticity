import pytest
import numpy as np

from model import pop_rate, simulate_population

def test_zero_population():
    assert pop_rate(0.0, 0.0, 1.0) == pytest.approx(0.0)

def test_below_capacity():
    assert pop_rate(0.0, 0.1, 1.0) == pytest.approx(0.09)

def test_at_capacity():
    assert pop_rate(0.0, 1.0, 1.0) == pytest.approx(0.0)

def test_above_capacity():
    assert pop_rate(0.0, 1.5, 1.0) == pytest.approx(-0.75)

def test_different_capacity():
    assert pop_rate(0.0, 2.0, 4.0) == pytest.approx(1.0)

def test_rate_does_not_depend_on_time():
    early_time = pop_rate(t=0.0, x=0.1, K=1.0)
    late_time = pop_rate(t=10.0, x=0.1, K=1.0)

    assert early_time == pytest.approx(late_time)

def test_rate_accepts_numpy_array():
    population = np.array([0.1])

    rate = pop_rate(t=0.0, x=population, K=1.0)

    assert rate.shape == population.shape
    np.testing.assert_allclose(rate, [0.09])

def tests_rate_handling_multiple_populations():
    populations = np.array([0.0,0.1,1.0,1.5])

    rates = pop_rate(t=0.0, x=populations, K=1.0)

    assert rates.shape == populations.shape

    np.testing.assert_allclose(
        rates,
        [0.0, 0.09, 0.0, -0.75],
        atol=1e-12
    )


def test_simulation_matches_exact_solution():
    K = 1.0
    x_initial = 0.1

    solution = simulate_population(x_initial=x_initial, K=K)

    assert solution.success, solution.message
    assert solution.t[-1] == pytest.approx(10.0)

    expected = K / (1 + (K / x_initial - 1) * np.exp(-solution.t))

   
    np.testing.assert_allclose(
        solution.y[0], expected, rtol=5e-3, atol=1e-6, equal_nan=False
    )
