"""Validate Ackland & Gallagher (2004), PRL 93, 158701, Eqs. (1)--(5).

Run from either the workspace or Plasticity with ``python -m pytest``.
The paper is https://doi.org/10.1103/PhysRevLett.93.158701.

For each (predator, prey) edge, A is the nonnegative prey-loss magnitude:
M[prey, predator] = -A and M[predator, prey] = b*A, as in Eqs. (1)--(2).
Applying Eq. (5) to that prey-loss coefficient gives
dA/dt = epsilon*(x_prey - x_predator)*A, with an upper limit of one.
We test this linked-coefficient convention, not two independently evolving
matrix entries. Equation (6) is a different model and is not tested here.

Analytical solutions, flux balances, equilibria and relabelling identities are
the oracles; production output is never used to generate expected values.
The paper's ensemble observations (chaos, power laws, eventual loss of cycles)
are not universal properties of every small web and are not unit assertions.
No extinction threshold is invented: the continuous ODE's zero-population
boundary is tested, rather than the paper's separate species-removal procedure.
Only physically admissible states/graphs are prescribed by these tests.
"""

from contextlib import ExitStack, redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import solve_ivp


PROJECT = Path(__file__).resolve().parents[1]


def load_module(filename):
    """Load the actual source by path, independently of pytest's working dir."""
    spec = importlib.util.spec_from_file_location(
        f"plasticity_test_{Path(filename).stem}", PROJECT / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_model = load_module("model.py")
pop_rate = legacy_model.pop_rate
simulate_population = legacy_model.simulate_population


@pytest.fixture(scope="module")
def script_run():
    """Import the real script without its 2000-time-unit run or GUI windows.

    Capture its solver configuration, returning the initial state ONLY for
    import-time plotting. Every trajectory assertion below uses real SciPy and
    the unmodified production RHS. No AST extraction or replacement RHS is used.
    """
    calls = []

    def capture_solve(fun, t_span, y0, **kwargs):
        calls.append((fun, t_span, np.array(y0, copy=True), kwargs))
        return SimpleNamespace(
            success=True, message="Integration deferred by test loader",
            t=np.array([t_span[0]]), y=np.asarray(y0)[:, None].copy(),
        )

    with ExitStack() as stack:
        stack.enter_context(patch("scipy.integrate.solve_ivp", capture_solve))
        for name in ("figure", "plot", "xlabel", "ylabel", "legend", "show"):
            stack.enter_context(patch(f"matplotlib.pyplot.{name}"))
        stack.enter_context(redirect_stdout(StringIO()))
        module = load_module("2004.py")

    assert len(calls) == 1, "Expected one simulation entry point in 2004.py"
    return SimpleNamespace(module=module, call=calls[0])


@pytest.fixture
def foodweb(script_run):
    return script_run.module


@pytest.fixture
def configure_web(foodweb, monkeypatch):
    """Install a controlled web and restore every production global afterwards."""
    def configure(autotrophs, edges, b=0.1):
        monkeypatch.setattr(foodweb, "N", len(autotrophs))
        monkeypatch.setattr(foodweb, "autotroph", np.array(autotrophs, dtype=bool))
        monkeypatch.setattr(foodweb, "edges", list(edges))
        monkeypatch.setattr(foodweb, "b", b)
        return foodweb.rates

    return configure


def integrate(rhs, y0, *, c=0.01, K=5.0, epsilon=0.01, duration=10.0,
              method="DOP853", rtol=1e-10, atol=1e-12, max_step=np.inf,
              samples=101):
    """Solve real trajectories; tolerances are intentionally tighter than asserts."""
    times = np.linspace(0.0, duration, samples)
    solution = solve_ivp(
        rhs, (0.0, duration), np.asarray(y0, dtype=float), args=(c, K, epsilon),
        method=method, rtol=rtol, atol=atol, max_step=max_step, t_eval=times,
    )
    assert solution.success, solution.message
    assert_allclose(solution.t, times, rtol=0, atol=0)
    assert solution.y.shape == (len(y0), samples)
    assert np.isfinite(solution.y).all()
    assert_allclose(solution.y[:, 0], y0, rtol=0, atol=0)
    return solution

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
