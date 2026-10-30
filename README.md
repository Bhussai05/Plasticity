# Plasticity

The food-web simulation from `2004.py`, split into source code, run scripts,
and YAML settings. Autotrophs have fixed environmental connections that
contribute competition, while biological feeding links evolve.

```text
Plasticity/
├── 2004.py                    # Compatibility entry point
├── configs/
│   └── 2004.yaml              # Simulation settings
├── scripts/
│   ├── run_2004.py            # Prompt and simulation loop
│   ├── bc_sweep.py            # Search over b and c
│   └── check_env.py           # Two-autotroph equation check
└── src/
    └── plasticity/
        ├── config.py         # YAML loading
        ├── model.py          # Web initialization and rates()
        └── plotting.py       # Diagnostics, figures, and saving
```

From the `Plasticity` directory:

```sh
python -m pip install -r requirements.txt
python scripts/run_2004.py
```

The original command, `python 2004.py`, also works. Both entry points find the
default configuration relative to the project, regardless of your working
directory.

Edit `configs/2004.yaml` to change the existing parameters, or copy it for a
separate experiment and select that file:

```sh
python scripts/run_2004.py --config configs/my_experiment.yaml
```

Answer `y` to run equations 5, 6, and 7, each with clipping on and off, using the
same initial web, and save the six main figures to `DESKTOP_PATH`. Answer `n`
(or press Enter) to run the configured `EVOLUTION_EQUATION` and `CLIP_LINKS`
and display the plots. The default end time remains 10000.

Use `.inf` for an unbounded `MAX_STEP`, `true`/`false` for clipping, and numeric
scientific notation such as `1.0e-6` for tolerances. Set `DESKTOP_PATH` to the
desired output directory.

Importing source modules does not prompt, run the solver, or create figures.
The web and simulation settings are passed to `rates()` as arguments.

Autotroph growth uses `H` (default `10.0`) for density dependence and
environmental competition, and applies the death rate `c` to autotrophs as
well as heterotrophs. For an autotroph `i`, the equation is:

```text
dx_i/dt = x_i * [(M @ x)_i + 1 - c - x_i/H
                - sum(j != i, E_j * x_j)/H]
```

Each autotroph has one fixed environmental strength `E_i`; heterotrophs have
zero environmental strength. These strengths are drawn after the biological
network to preserve its random draw order. They are returned separately from
`initialize_web()` and remain outside the integrated state, which contains
only biological populations followed by evolving biological link strengths.

Check both the standard and Numba implementations before a full simulation:

```sh
python scripts/check_env.py
```

Both should print population derivatives `[0.78 1.38]`. The parameter sweep
reuses one environment vector across all its trials:

```sh
python scripts/bc_sweep.py
```

Rerun the sweep after this model change because `c` now also affects autotrophs.

The existing `tests/test_model.py` targets an older implementation and imports
the missing root `model.py`; it does not validate the current implementation.
