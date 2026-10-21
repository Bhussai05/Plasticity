# Plasticity

The procedural simulation from `2004.py`, split into source code, a run script,
and YAML settings. The equations, initialization order, solver settings,
diagnostics, plots, and six-run sequence are unchanged.

```text
Plasticity/
├── 2004.py                    # Compatibility entry point
├── configs/
│   └── 2004.yaml              # Existing simulation constants
├── scripts/
│   └── run_2004.py            # Prompt and simulation loop
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

YAML retains the original variable names and values. Use `.inf` for an
unbounded `MAX_STEP`, `true`/`false` for clipping, and numeric scientific
notation such as `1.0e-6` for tolerances. Set `DESKTOP_PATH` to the desired
output directory.

Importing source modules does not prompt, run the solver, or create figures.
`rates()` has the same body as before; the web and constants that were globals
are now passed as arguments.

The existing `tests/test_model.py` targets an older implementation and imports
the already-missing root `model.py`. Those tests are left unchanged by this
refactor.
