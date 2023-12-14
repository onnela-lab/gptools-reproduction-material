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

# Linear regression example from Section 2 of the manuscript

```{code-cell} ipython3
import numpy as np

np.random.seed(0)
n = 100
p = 3
X = np.random.normal(0, 1, (n, p))
theta = np.random.normal(0, 1, p)
sigma = np.random.gamma(2, 2)
y = np.random.normal(X @ theta, sigma)

print(f"coefficients: {theta}")
print(f"observation noise scale: {sigma}")
```

```{code-cell} ipython3
import cmdstanpy

model = cmdstanpy.CmdStanModel(stan_file="linear.stan")
fit = model.sample(data={"n": n, "p": p, "X": X, "y": y}, seed=0)

print(fit.summary())
```
