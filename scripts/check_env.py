from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.model import rates
from bc_sweep import fast_rates

autotroph = np.array([True, True])
edges = np.empty((0, 2), dtype=np.int64)
env_strengths = np.array([0.2, 0.7])
y = np.array([2.0, 3.0])

# c, H, epsilon, equation, clip_links, N, b, gamma,
# threshold, autotroph, edges, env_strengths
args = (
    0.2, 10.0, 0.001, 5, False, 2, 0.32571, 0.0,
    1e-6, autotroph, edges, env_strengths,
)

expected = np.array([
    2.0 * (1.0 - 0.2 - 2.0 / 10.0 - 0.7 * 3.0 / 10.0),
    3.0 * (1.0 - 0.2 - 3.0 / 10.0 - 0.2 * 2.0 / 10.0),
])

for rhs in (rates, fast_rates):
    actual = rhs(0.0, y, *args)
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-14,
    )
    print(rhs.__name__, actual)