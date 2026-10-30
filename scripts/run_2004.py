"""Run the original procedural simulation using settings from YAML."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from scipy.integrate import solve_ivp


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plasticity.config import DEFAULT_CONFIG, load_config
from plasticity.model import initialize_web, rates
from plasticity.plotting import plot_results


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

    for equation, clip_links in runs:
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
            method=config["METHOD"],
            rtol=config["RTOL"],
            atol=config["ATOL"],
            max_step=config["MAX_STEP"],
        )

        plot_results(
            sol, config, autotroph, edges, equation, clip_links,
            generate_all, run_stamp,
        )


if __name__ == "__main__":
    main()
