

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


# Ackland & Gallagher (2004), https://doi.org/10.1103/PhysRevLett.93.158701.
# Equations (3)--(5): local dynamics on deliberately constructed food webs.
# A = -M[prey,predator]; its linked predator-gain coefficient is b*A (Eq. 2).

def test_generated_web_is_a_valid_trophic_initial_condition(foodweb):
    n = foodweb.N
    edges = foodweb.edges
    assert foodweb.autotroph.shape == (n,)
    assert np.count_nonzero(foodweb.autotroph) == foodweb.n_autotrophs
    assert foodweb.y_0.shape == (n + len(edges),)
    assert np.isfinite(foodweb.y_0).all()
    assert (foodweb.y_0[:n] >= 0).all()
    assert ((foodweb.y_0[n:] >= 0) & (foodweb.y_0[n:] <= 1)).all()
    assert len({frozenset(edge) for edge in edges}) == len(edges)
    for predator, prey in edges:
        assert 0 <= predator < n and 0 <= prey < n
        assert predator != prey
        assert not foodweb.autotroph[predator]


@pytest.mark.parametrize("K", [0.25, 1.0, 7.0])
@pytest.mark.parametrize("fraction", [0.0, 0.2, 1.0, 2.0])
def test_isolated_autotroph_obeys_logistic_equation(configure_web, K, fraction):
    rhs = configure_web([True], [])
    x = K * fraction
    assert_allclose(rhs(0, np.array([x]), 0.3, K, 0.7),
                    [x * (1 - fraction)], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("c", [0.0, 0.01, 0.6])
def test_isolated_heterotroph_has_only_mortality(configure_web, c):
    rhs = configure_web([False, False], [])
    assert_allclose(rhs(0, np.array([0.4, 9.0]), c, 2, 0.2),
                    [-0.4*c, -9*c], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("b", [0.0, 0.1, 0.7, 1.0])
def test_single_edge_has_correct_loss_conversion_and_feedback(configure_web, b):
    rhs = configure_web([True, False], [(1, 0)], b=b)
    # Feeding removes .4*2*3 = 2.4 prey and produces b*2.4 predators.
    actual = rhs(0, np.array([2.0, 3.0, 0.4]), 0.1, 10, 0.05)
    assert_allclose(actual, [-0.8, 2.4*b - 0.3, -0.02], atol=1e-14)


def test_mixed_web_sums_all_incoming_and_outgoing_fluxes(configure_web):
    rhs = configure_web([True, False, True, False, False],
                        [(1, 0), (3, 1), (3, 2), (4, 3)], b=0.2)
    # Fluxes are 2.4, 3, 6 and 3.5; species 3 both eats and is eaten.
    actual = rhs(0, np.array([2., 3., 4., 5., 7., .4, .2, .3, .1]),
                 0.1, 10, 0.05)
    assert_allclose(actual, [-.8, -2.82, -3.6, -2.2, 0.,
                             -.02, -.02, -.015, -.01], atol=2e-14)


def test_several_predators_share_one_prey(configure_web):
    rhs = configure_web([True, False, False], [(1, 0), (2, 0)], b=0.25)
    actual = rhs(0, np.array([4., 2., 3., .2, .5]), 0.1, 8, 0.1)
    assert_allclose(actual, [-5.6, .2, 1.2, .04, .05], atol=1e-14)


def test_trophic_cycle_is_supported_without_imposing_a_hierarchy(configure_web):
    rhs = configure_web([False]*3, [(0, 1), (1, 2), (2, 0)], b=0.25)
    actual = rhs(0, np.array([2., 3., 5., .2, .4, .1]), 0.1, 5, 0.02)
    assert_allclose(actual, [-.9, 0., -6.25, .004, .016, -.006], atol=1e-14)


@pytest.mark.parametrize("prey,predator,alpha,expected", [
    (4., 1., .3, .06),       # prey abundance strengthens feeding
    (1., 4., .3, -.06),      # predator abundance weakens feeding
    (2., 2., .3, 0.),        # equal abundance: no relative adaptation
    (4., 1., 0., 0.),        # absent link stays absent
    (4., 1., 1., 0.),        # outward flow at the upper cap is suppressed
    (1., 4., 1., -.2),       # inward flow must still leave the upper cap
    (2., 2., 1., 0.),
    (4., 1., .999, .1998),   # adaptation must not stop before the cap
    (4., 1., 1e-12, 2e-13), # weak links must not be thresholded away
])
def test_adaptation_direction_and_boundaries(
        configure_web, prey, predator, alpha, expected):
    rhs = configure_web([True, False], [(1, 0)])
    actual = rhs(0, np.array([prey, predator, alpha]), .01, 5, 1/15)
    assert actual[2] == pytest.approx(expected, rel=1e-12, abs=1e-25)


@pytest.mark.parametrize("extinct", range(4))
def test_extinct_species_cannot_be_recreated_by_feeding(configure_web, extinct):
    rhs = configure_web([True, False, False, False],
                        [(1, 0), (2, 1), (3, 1), (3, 2)])
    state = np.array([2., 3., 4., 5., .2, .3, .4, .5])
    state[extinct] = 0
    assert rhs(0, state, .01, 5, .1)[extinct] == 0


def test_zero_strength_link_is_dynamically_identical_to_no_link(configure_web):
    state = np.array([2., 3., 4., .3, 0.])
    rhs = configure_web([True, False, False], [(1, 0), (2, 1)])
    with_zero_link = rhs(0, state, .01, 5, .1)
    rhs = configure_web([True, False, False], [(1, 0)])
    without_link = rhs(0, state[:-1], .01, 5, .1)
    assert_allclose(with_zero_link[:-1], without_link, rtol=0, atol=0)
    assert with_zero_link[-1] == 0


def test_epsilon_only_scales_link_derivatives(configure_web):
    rhs = configure_web([True, False, False], [(1, 0), (2, 1)])
    state = np.array([3., 1., 2., .2, .4])
    frozen = rhs(0, state, .01, 5, 0.)
    slow = rhs(0, state, .01, 5, .03)
    fast = rhs(0, state, .01, 5, .12)
    assert_allclose(frozen[3:], 0, atol=0)
    assert_allclose(slow[:3], frozen[:3], rtol=0, atol=0)
    assert_allclose(fast[:3], frozen[:3], rtol=0, atol=0)
    assert_allclose(fast[3:], 4*slow[3:], rtol=1e-14, atol=1e-15)


def test_capacity_and_mortality_affect_only_the_correct_species(configure_web):
    rhs = configure_web([True, False, True], [(1, 0), (1, 2)])
    state = np.array([2., 3., 4., .2, .4])
    base = rhs(0, state, .1, 5, .03)
    changed_c = rhs(0, state, .3, 5, .03)
    changed_K = rhs(0, state, .1, 10, .03)
    assert_allclose(changed_c - base, [0., -.6, 0., 0., 0.], atol=1e-14)
    assert_allclose(changed_K - base, [.4, 0., 1.6, 0., 0.], atol=1e-14)


def test_rhs_is_autonomous_and_does_not_mutate_inputs(configure_web, foodweb):
    rhs = configure_web([True, False, False], [(1, 0), (2, 1)])
    state = np.array([3., 1., 2., .2, .4])
    before = state.copy()
    mask = foodweb.autotroph.copy()
    edges = foodweb.edges.copy()
    state.flags.writeable = False
    first = rhs(0, state, .01, 5, .03)
    assert first.shape == state.shape
    assert np.isfinite(first).all()
    assert not np.shares_memory(first, state)
    assert_allclose(rhs(1234, state, .01, 5, .03), first, rtol=0, atol=0)
    assert_allclose(state, before, rtol=0, atol=0)
    assert np.array_equal(foodweb.autotroph, mask)
    assert foodweb.edges == edges


# Structural properties that do not depend on a particular seeded simulation.

@pytest.mark.parametrize("b", [0., .1, .65, 1.])
@pytest.mark.parametrize("seed", [7, 19, 91])
def test_total_biomass_budget_on_varied_webs(configure_web, b, seed):
    rng = np.random.default_rng(seed)
    n = 8
    mask = np.array([True, False, True, False, False, False, True, False])
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (mask[i] and mask[j]) or rng.random() > .6:
                continue
            if mask[i]:
                edges.append((j, i))
            elif mask[j] or rng.random() < .5:
                edges.append((i, j))
            else:
                edges.append((j, i))
    x = rng.uniform(.1, 8, n)
    alpha = rng.uniform(.01, .99, len(edges))
    rhs = configure_web(mask, edges, b=b)
    dx = rhs(0, np.r_[x, alpha], .07, 3, .02)[:n]
    # Every feeding event loses (1-b) units; birth and mortality are external.
    total_feeding = sum(a*x[pred]*x[prey] for a, (pred, prey) in zip(alpha, edges))
    external = sum(x[mask]*(1 - x[mask]/3)) - .07*sum(x[~mask])
    assert sum(dx) == pytest.approx(external - (1-b)*total_feeding,
                                    rel=1e-12, abs=1e-12)


def test_species_relabelling_preserves_dynamics(configure_web):
    mask = np.array([True, False, False, True, False])
    edges = [(1, 0), (2, 1), (4, 3), (1, 4)]
    x = np.array([2., 4., 1., 3., 5.])
    alpha = np.array([.2, .7, .4, .1])
    rhs = configure_web(mask, edges, b=.3)
    original = rhs(0, np.r_[x, alpha], .05, 6, .02)
    permutation = np.array([3, 1, 4, 0, 2])  # new index -> old index
    inverse = np.argsort(permutation)
    new_edges = [(inverse[pred], inverse[prey]) for pred, prey in edges]
    rhs = configure_web(mask[permutation], new_edges, b=.3)
    relabelled = rhs(0, np.r_[x[permutation], alpha], .05, 6, .02)
    assert_allclose(relabelled, np.r_[original[:5][permutation], original[5:]],
                    rtol=1e-13, atol=1e-14)


def test_edge_reordering_only_reorders_link_derivatives(configure_web):
    edges = [(1, 0), (2, 1), (2, 0)]
    x, alpha = np.array([2., 4., 3.]), np.array([.2, .7, .4])
    rhs = configure_web([True, False, False], edges, b=.3)
    original = rhs(0, np.r_[x, alpha], .05, 6, .02)
    order = [2, 0, 1]
    rhs = configure_web([True, False, False], [edges[i] for i in order], b=.3)
    reordered = rhs(0, np.r_[x, alpha[order]], .05, 6, .02)
    assert_allclose(reordered, np.r_[original[:3], original[3:][order]], atol=1e-14)


def test_disconnected_component_cannot_influence_other_component(configure_web):
    rhs = configure_web([True, False, True, False], [(1, 0), (3, 2)])
    state = np.array([2., 3., 4., 5., .2, .8])
    first = rhs(0, state, .01, 5, .04)
    state[[2, 3, 5]] = [100., 50., .01]
    second = rhs(0, state, .01, 5, .04)
    assert_allclose(first[[0, 1, 4]], second[[0, 1, 4]], rtol=0, atol=0)


# Numerical constraints must hold in raw solver states, before plotting clips A.

@pytest.fixture
def cap_release_solution(configure_web, script_run):
    """Use the script's actual method/tolerances/step policy on a solvable case.

    For b=1,c=0, prey+predator=S=4. In the interior,
    A-epsilon*log(prey*predator) is constant. Starting at (3,1,.9)
    with epsilon=.5 would reach .9+.5*log(4/3)>1 at equal populations,
    so the cap MUST be contacted first. The trajectory leaves the cap at
    prey=predator=2. Thereafter A-.5*log(prey*predator/4)=1 exactly.
    This tests consequences of hitting the cap, not just a rounded plot.
    """
    rhs = configure_web([False, False], [(1, 0)], b=1.)
    options = dict(script_run.call[3])
    options.update(args=(0., 5., .5), t_eval=np.linspace(0., 10., 501))
    solution = solve_ivp(rhs, (0., 10.), [3., 1., .9], **options)
    assert solution.success, solution.message
    assert solution.t[-1] == 10.
    assert np.isfinite(solution.y).all()
    assert_allclose(solution.y[:2].sum(axis=0), 4., rtol=0, atol=1e-10)
    return solution


def test_script_solver_respects_raw_link_upper_bound(cap_release_solution):
    # Allow 0.5% numerical slack (5x the script's default relative tolerance),
    # so failure is not a demand for machine precision from default RK45.
    maximum = cap_release_solution.y[2].max()
    assert maximum <= 1.005, (
        f"Raw link strength reached {maximum:.9g}; the paper caps it at 1. "
        "Clipping A_solution after integration cannot repair this trajectory."
    )


def test_script_solver_preserves_exact_identity_after_cap_release(cap_release_solution):
    prey, predator, alpha = cap_release_solution.y
    released = (prey < predator) & (alpha < .95)
    assert released.any(), "This trajectory must contact and then leave the cap"
    invariant = alpha[released] - .5*np.log(prey[released]*predator[released]/4.)
    assert_allclose(
        invariant, 1., rtol=0, atol=.005,
        err_msg="Cap overshoot changes subsequent ecological dynamics",
    )


@pytest.mark.parametrize("alpha0", [0., .2, .8, 1.])
def test_exact_link_growth_and_saturation_with_absent_predator(configure_web, alpha0):
    rhs = configure_web([True, False], [(1, 0)])
    solution = integrate(rhs, [5., 0., alpha0], duration=40., max_step=.1)
    # Prey stays at K, so the growth exponent is epsilon*K*t until contact.
    expected_alpha = np.minimum(alpha0*np.exp(.05*solution.t), 1.)
    assert_allclose(solution.y[0], 5., rtol=0, atol=1e-12)
    assert_allclose(solution.y[1], 0., rtol=0, atol=1e-12)
    assert_allclose(solution.y[2], expected_alpha, rtol=2e-8, atol=2e-10)


# Independent analytical oracles from Ackland et al. Eqs. (3)-(5).


@pytest.mark.parametrize("carrying_capacity", [0.25, 5.0, 30.0])
def test_isolated_autotrophs_follow_exact_logistic_solution(
    configure_web, carrying_capacity
):
    rhs = configure_web([True] * 4, [])
    initial = carrying_capacity * np.array([0.0, 0.04, 1.0, 2.3])
    result = integrate(rhs, initial, K=carrying_capacity, c=0.31,
                       epsilon=0.2, duration=18.0)
    expected = (
        carrying_capacity * initial[:, None]
        / (initial[:, None]
           + (carrying_capacity - initial[:, None]) * np.exp(-result.t))
    )
    assert_allclose(result.y, expected, rtol=3e-9, atol=3e-11)


@pytest.mark.parametrize("death_rate", [0.0, 0.007, 0.3])
def test_isolated_heterotrophs_follow_exact_exponential_mortality(
    configure_web, death_rate
):
    rhs = configure_web([False] * 4, [])
    initial = np.array([0.0, 0.02, 1.5, 8.0])
    result = integrate(rhs, initial, c=death_rate, K=0.25,
                       epsilon=0.17, duration=30.0)
    expected = initial[:, None] * np.exp(-death_rate * result.t)
    assert_allclose(result.y, expected, rtol=3e-9, atol=3e-11)


@pytest.mark.parametrize("death_rate", [0.0, 0.07])
def test_starving_predator_and_its_adaptive_link_have_exact_trajectories(
    configure_web, death_rate
):
    # An extinct prey remains extinct. The predator decays exponentially,
    # while d(log A)/dt = -epsilon * predator(t).
    rhs = configure_web([True, False], [(1, 0)], b=0.37)
    predator0, strength0, adaptation = 2.5, 0.6, 0.05
    result = integrate(rhs, [0.0, predator0, strength0], c=death_rate,
                       epsilon=adaptation, duration=20.0)
    predator = predator0 * np.exp(-death_rate * result.t)
    predator_integral = (
        predator0 * result.t if death_rate == 0.0 else
        -predator0 * np.expm1(-death_rate * result.t) / death_rate
    )
    strength = strength0 * np.exp(-adaptation * predator_integral)
    assert_allclose(result.y[0], 0.0, atol=1e-13)
    assert_allclose(result.y[1], predator, rtol=3e-9, atol=3e-11)
    assert_allclose(result.y[2], strength, rtol=3e-9, atol=3e-11)


@pytest.mark.parametrize("prey0", [0.4, 3.0, 7.0])
def test_absent_predator_leaves_exact_logistic_prey_and_link_growth(
    configure_web, prey0
):
    # Integral_0^t prey(s) ds = K log(1 + (prey0/K)(exp(t)-1)).
    # Eq. (5) therefore has an exact solution even as the prey changes.
    rhs = configure_web([True, False], [(1, 0)], b=0.22)
    capacity, adaptation, strength0 = 3.0, 0.02, 0.08
    result = integrate(rhs, [prey0, 0.0, strength0], K=capacity,
                       c=0.13, epsilon=adaptation, duration=8.0)
    prey = capacity * prey0 / (
        prey0 + (capacity - prey0) * np.exp(-result.t)
    )
    prey_integral = capacity * np.log1p(
        (prey0 / capacity) * np.expm1(result.t)
    )
    strength = strength0 * np.exp(adaptation * prey_integral)
    assert np.max(strength) < 1.0  # Required for the uncapped exact solution.
    assert_allclose(result.y[0], prey, rtol=3e-9, atol=3e-11)
    assert_allclose(result.y[1], 0.0, atol=1e-13)
    assert_allclose(result.y[2], strength, rtol=3e-9, atol=3e-11)


@pytest.mark.parametrize(
    "capacity, efficiency, death_rate, strength",
    [(5.0, 0.3, 0.18, 0.4), (2.0, 0.1, 0.025, 0.25),
     (9.0, 0.65, 0.12, 0.5)],
)
def test_fixed_link_coexistence_equilibrium_remains_stationary(
    configure_web, capacity, efficiency, death_rate, strength
):
    # Eq. (4): prey*=c/(b A). Eq. (3): predator*=(1-prey*/K)/A.
    rhs = configure_web([True, False], [(1, 0)], b=efficiency)
    prey = death_rate / (efficiency * strength)
    predator = (1.0 - prey / capacity) / strength
    initial = np.array([prey, predator, strength])
    assert predator > 0.0
    result = integrate(rhs, initial, K=capacity, c=death_rate,
                       epsilon=0.0, duration=100.0)
    assert_allclose(result.y, np.repeat(initial[:, None], result.t.size, axis=1),
                    rtol=1e-9, atol=2e-11)


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_fixed_link_trajectory_dissipates_lyapunov_function_and_converges(
    configure_web, method
):
    # V=b[x-x*-x*log(x/x*)]+[p-p*-p*log(p/p*)] obeys
    # dV/dt=-(b/K)(x-x*)**2. This checks a nonstationary coupled trajectory,
    # including its approach to the analytic coexistence equilibrium.
    efficiency, capacity, death_rate, strength = 0.3, 5.0, 0.15, 0.4
    rhs = configure_web([True, False], [(1, 0)], b=efficiency)
    prey_star = death_rate / (efficiency * strength)
    predator_star = (1.0 - prey_star / capacity) / strength
    result = integrate(rhs, [0.6, 2.6, strength], c=death_rate, K=capacity,
                       epsilon=0.0, duration=160.0, method=method, samples=801)
    prey, predator, link = result.y
    assert np.all(prey > 0.0)
    assert np.all(predator > 0.0)
    lyapunov = efficiency * (
        prey - prey_star - prey_star * np.log(prey / prey_star)
    ) + predator - predator_star - predator_star * np.log(predator / predator_star)
    assert np.max(np.diff(lyapunov)) < 5e-10
    assert lyapunov[-1] < lyapunov[0] * 1e-8
    assert_allclose(result.y[:2, -1], [prey_star, predator_star],
                    rtol=3e-7, atol=2e-9)
    assert_allclose(link, strength, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "efficiency, death_rate, capacity",
    [(0.4, 0.08, 3.0), (0.1, 0.01, 5.0), (0.7, 0.14, 2.0)],
)
def test_positive_adaptive_coexistence_equilibrium_remains_stationary(
    configure_web, efficiency, death_rate, capacity
):
    # A positive interior Eq. (5) equilibrium requires prey=predator.
    # Combining this with Eqs. (3)-(4) fixes both population and strength.
    rhs = configure_web([True, False], [(1, 0)], b=efficiency)
    population = capacity * (1.0 - death_rate / efficiency)
    strength = death_rate / (efficiency * population)
    assert 0.0 < strength < 1.0
    initial = np.array([population, population, strength])
    assert_allclose(rhs(0., initial, death_rate, capacity, .07),
                    0., rtol=0, atol=1e-13)
    # Bound RK steps so a near-zero RHS cannot invite huge, unstable steps.
    result = integrate(rhs, initial, K=capacity, c=death_rate,
                       epsilon=0.07, duration=50.0, max_step=1.0)
    assert_allclose(result.y, np.repeat(initial[:, None], result.t.size, axis=1),
                    rtol=2e-9, atol=3e-11)


@pytest.mark.parametrize("death_rate", [0.0, 0.035])
def test_unit_efficiency_heterotroph_network_has_exact_total_biomass(
    configure_web, death_rate
):
    # At b=1 every consumption loss is another species' equal gain, so the
    # entire network obeys d(sum x)/dt=-c sum x even with evolving links.
    edges = [(1, 0), (2, 1), (0, 2), (3, 0), (3, 2)]
    rhs = configure_web([False] * 4, edges, b=1.0)
    initial = np.array([0.8, 1.1, 0.7, 0.6, 0.05, 0.08, 0.11, 0.07, 0.03])
    result = integrate(rhs, initial, c=death_rate,
                       epsilon=0.01, duration=20.0)
    expected = initial[:4].sum() * np.exp(-death_rate * result.t)
    assert_allclose(result.y[:4].sum(axis=0), expected,
                    rtol=3e-10, atol=3e-11)
    assert np.ptp(result.y[0]) > 0.1


@pytest.mark.parametrize("efficiency, adaptation", [(0.5, 0.05), (0.2, 0.015), (0.8, 0.03)])
def test_two_heterotrophs_preserve_independent_adaptive_first_integrals(
    configure_web, efficiency, adaptation
):
    # For c=0, P=prey+predator/b and
    # H=A-(epsilon/b)log(predator)-epsilon log(prey) are both constant.
    # The second invariant couples Eq. (5) to both population equations.
    rhs = configure_web([False, False], [(1, 0)], b=efficiency)
    initial = np.array([3.0, 1.0, 0.1])
    result = integrate(rhs, initial, c=0.0,
                       epsilon=adaptation, duration=15.0)
    prey, predator, strength = result.y
    assert np.all(prey > 0.0)
    assert np.all(predator > 0.0)
    assert np.all((strength > 0.0) & (strength < 1.0))
    weighted_biomass = prey + predator / efficiency
    first_integral = (strength - (adaptation / efficiency) * np.log(predator)
                      - adaptation * np.log(prey))
    expected_biomass = initial[0] + initial[1] / efficiency
    expected_integral = (initial[2] - (adaptation / efficiency) * np.log(initial[1])
                         - adaptation * np.log(initial[0]))
    assert_allclose(weighted_biomass, expected_biomass, rtol=3e-10, atol=3e-11)
    assert_allclose(first_integral, expected_integral, rtol=0.0, atol=2e-9)
    assert np.ptp(prey) > 0.5
    assert np.ptp(strength) > 0.005


@pytest.mark.parametrize("efficiency, adaptation", [(0.1, 0.03), (0.7, 0.2)])
def test_trophic_cycle_preserves_product_of_interior_link_strengths(
    configure_web, efficiency, adaptation
):
    # Summing Eq. (5) as logarithmic derivatives around a directed cycle
    # telescopes to zero, irrespective of mortality or birth efficiency.
    rhs = configure_web([False] * 3, [(1, 0), (2, 1), (0, 2)], b=efficiency)
    initial = np.array([1.4, 0.8, 0.5, 0.07, 0.11, 0.09])
    result = integrate(rhs, initial, c=0.03,
                       epsilon=adaptation, duration=20.0)
    strengths = result.y[3:]
    assert np.all((strengths > 0.0) & (strengths < 1.0))
    assert_allclose(np.log(strengths).sum(axis=0), np.log(initial[3:]).sum(),
                    rtol=0.0, atol=2e-9)
    assert np.ptp(strengths[0]) > 0.005
