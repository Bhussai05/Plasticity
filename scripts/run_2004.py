"""Run the original procedural simulation using settings from YAML."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, Radau, solve_ivp
from tqdm import tqdm


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plasticity.config import DEFAULT_CONFIG, load_config
from plasticity.model import initialize_web, rates
from plasticity.plotting import plot_results


SOLVER_TYPES = {
    "RK23": RK23,
    "RK45": RK45,
    "DOP853": DOP853,
    "Radau": Radau,
    "BDF": BDF,
    "LSODA": LSODA,
}


def _solver_with_progress(method, progress):
    """Report accepted steps while preserving the selected solver's behavior."""
    if method not in SOLVER_TYPES:
        raise ValueError(f"Unsupported solver method: {method}")
    solver_type = SOLVER_TYPES[method]

    class ProgressSolver(solver_type):
        def step(self):
            previous_time = self.t
            message = super().step()
            if self.status != "failed":
                progress.update(abs(self.t - previous_time))
            return message

    return ProgressSolver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML settings file (default: configs/2004.yaml).",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    autotroph, edges, y_0, env_strengths = initialize_web(
        config["seed"], config["N"], config["connection"], config["n_autotrophs"], config["b"],
    )

    answer = input(
        "Generate all 6 figures and save them to Desktop? [y/N]: "
    )
    generate_all = answer.strip().lower() in ("y", "yes")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if generate_all:
        runs = [
            (5, True),
            (5, False),
            (6, True),
            (6, False),
            (7, True),
            (7, False),
        ]
        config["DESKTOP_PATH"].mkdir(parents=True, exist_ok=True)
    else:
        runs = [(config["EVOLUTION_EQUATION"], config["CLIP_LINKS"])]

    with tqdm(
        runs, desc="Simulation runs", unit="run",
        disable=not generate_all, position=0, dynamic_ncols=True,
    ) as run_progress:
        for equation, clip_links in run_progress:
            with tqdm(
                total=abs(config["T_END"]),
                desc=f"Mode {equation}, clipping {'on' if clip_links else 'off'}",
                unit="sim-t", unit_scale=True, dynamic_ncols=True,
                position=1 if generate_all else 0, leave=not generate_all,
            ) as integration_progress:
                sol = solve_ivp(
                    rates,
                    (0.0, config["T_END"]),
                    y_0.copy(),
                    args=(
                        config["c"], config["H"], config["epsilon"],
                        equation, clip_links,
                        config["N"], config["b"], config["gamma"],
                        config["EXTINCTION_THRESHOLD"], autotroph, edges, env_strengths,
                    ),
                    method=_solver_with_progress(config["METHOD"], integration_progress),
                    rtol=config["RTOL"],
                    atol=config["ATOL"],
                    max_step=config["MAX_STEP"],
                )

            species_summary = None
            if sol.success:
                alive = sol.y[:config["N"], -1] > config["EXTINCTION_THRESHOLD"]
                surviving_autotrophs = int((alive & autotroph).sum())
                surviving_heterotrophs = int((alive & ~autotroph).sum())
                if surviving_heterotrophs:
                    ratio = f"{surviving_autotrophs / surviving_heterotrophs:.6g}"
                elif surviving_autotrophs:
                    ratio = "inf; no surviving heterotrophs"
                else:
                    ratio = "undefined; no surviving species"

                species_summary = (
                    f"Surviving autotrophs: {surviving_autotrophs}; "
                    f"heterotrophs: {surviving_heterotrophs}\n"
                    f"Autotroph:heterotroph ratio: "
                    f"{surviving_autotrophs}:{surviving_heterotrophs} ({ratio})"
                )

            with tqdm.external_write_mode(file=sys.stdout):
                plot_results(
                    sol, config, autotroph, edges, equation, clip_links,
                    generate_all, run_stamp, species_summary=species_summary,
                )
                sys.stdout.flush()


if __name__ == "__main__":
    main()
