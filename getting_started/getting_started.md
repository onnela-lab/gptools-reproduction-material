---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
>>> import cmdstanpy
>>> from gptools.stan import get_include
>>>
>>> model = cmdstanpy.CmdStanModel(
...     stan_file="getting_started.stan",
...     stanc_options={"include-paths": get_include()},
... )
>>> fit = model.sample(
...     data = {"n": 100, "sigma": 1, "length_scale": 0.1, "period": 1},
...     chains=1,
...     iter_warmup=500,
...     iter_sampling=50,
... )
>>> fit.f.shape
```

Expected output: `(50, 100)`
