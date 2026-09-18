import pytest
import numpy as np

from model import pop_rate, simulate_population

def test_zero_population():
    assert pop_rate(0.0, state=[0.0,0.0], K=1.0, a=1.0, b=0.1, c=0.01) == pytest.approx([0.0,0.0])

def test_below_capacity():
    assert pop_rate(0.0, state=[0.1,0.0], K=1.0, a=1.0, b=0.1, c=0.01) == pytest.approx([0.09,0.0])

def test_at_capacity():
    assert pop_rate(0.0, state=[1.0,0.0], K=1.0, a=1.0, b=0.1, c=0.01) == pytest.approx([0.0,0.0])

def test_above_capacity():
    assert pop_rate(0.0, state=[1.5,0.0], K=1.0, a=1.0, b=0.1, c=0.01) == pytest.approx([-0.75,0.0])

def test_different_capacity():
    assert pop_rate(0.0, state=[2.0,0.0], K=4.0, a=1.0, b=0.1, c=0.01) == pytest.approx([1.0,0.0])

def test_rate_does_not_depend_on_time():
    early_time = pop_rate(t=0.0, state=[0.1,0.5], K=1.0, a=1.0, b=0.1, c=0.01)
    late_time = pop_rate(t=10.0, state=[0.1,0.5], K=1.0, a=1.0, b=0.1, c=0.01)

    assert early_time == pytest.approx(late_time)

def test_rate_accepts_numpy_array():
    population = np.array([0.1, 0.0])

    rate = pop_rate(t=0.0, state=population, K=1.0, a=1.0, b=0.1, c=0.01)

    assert np.shape(rate) == population.shape
    np.testing.assert_allclose(rate, [0.09, 0.0])

def test_rate_handling_both_populations():
    populations = np.array([2.0, 3.0])

    rates = pop_rate(t=0.0, state=populations, K=10.0, a=0.2, b=0.3, c=0.1)

    assert np.shape(rates) == populations.shape

    np.testing.assert_allclose(
        rates,
        [0.4, 0.06],
        atol=1e-12
    )


def test_simulation_matches_exact_solution():
    K = 1.0
    x_initial = 0.1

    solution = simulate_population(
        state=[x_initial, 0.0], K=K, a=1.0, b=0.1, c=0.01
    )

    assert solution.success, solution.message
    assert solution.t[-1] == pytest.approx(10.0)

    expected = K / (1 + (K / x_initial - 1) * np.exp(-solution.t))

   
    np.testing.assert_allclose(
        solution.y[0], expected, rtol=5e-3, atol=1e-6, equal_nan=False
    )
    np.testing.assert_allclose(solution.y[1], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "state, expected",
    [
        ([0.0, 0.0], [0.0, 0.0]),
        ([2.0, 0.0], [1.6, 0.0]),
        ([0.0, 3.0], [0.0, -0.3]),
    ],
)
def test_population_rates(state, expected):
    rate = pop_rate(
        t=0.0,
        state=state,
        K=10.0,
        a=0.2,
        b=0.3,
        c=0.1,
    )

    assert rate == pytest.approx(expected)
