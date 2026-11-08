This is readme.

## Comparing per-beta runtimes

Compare saved solutions with:

```bash
python main.py compare --scenario-root scenarios_new --methods cbf mep_cbf_pruned slsqp
```

In addition to the usual comparison plots, this writes
`scenarios_new/comparison_metric_plots/beta_runtime_comparison.png`. Its three
panels show the first, middle, and last scheduled beta values (the default
schedule uses 1e-6, 1e-3, and 1). Each point is the mean across available
scenarios of the elapsed wall time for that beta stage, including its setup.
These are per-stage times, not cumulative runtimes, and a recorded time does
not establish that the solver converged.

The comparison summary JSON records the selected betas, means, and sample
counts. Solutions missing timing data, or using different selected betas, are
excluded with a warning. A timed-out run contributes only the beta stages it
actually reached. If no timing data is available, no beta-runtime plot is
created; existing solution files with `b_arr` and `compute_time_per_beta` can
be compared without re-solving.

Comparison calculations, plots, and their beta-runtime tests live in
`comparison.py`. Run those tests from this directory with:

```bash
python -m unittest comparison
```
