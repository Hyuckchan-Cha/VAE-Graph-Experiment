# VAE-graph-experiment

A simulation experiment on what happens when an item response model defines the latent trait **θ explicitly** (estimated per person by a VAE encoder) while also estimating a **graph S of direct item-to-item effects**.

> **Ongoing research.** This notebook is a tool for checking a hypothesis, not a finished study. It does not propose a new method or claim a conclusion. Each condition uses a single simulated data set and seed, so the figures show the direction of an effect, not its size or reproducibility.

## The question

In a VAE-based IRT model with an added item graph (called **VIF-model** here), the change in item *j*'s logit when item *k* flips from 0 to 1 travels along two paths:

- the **θ path**: item *k* changes the encoder's θ, which changes item *j* through its loading *a_j* (including a **self term**, where an item's own response feeds back into its own logit);
- the **S path**: the direct edge *s_jk*.

The data only pin down the total. The hypothesis being checked is that, with explicit θ, the split between the two paths is poorly constrained, so *A* and *S* can trade off against each other, especially when the graph is dense. Models that integrate θ out, such as FLaG-IRT (Chen et al., 2018), have no self-term path.

VIF-model is used here as a **probe** of this behaviour, not as a proposed model.

## What the notebook does

1. Simulates 24-item testlet data (N = 2,000) from three data-generating processes, each with a sparse and a dense true graph:
   - **VIF-type**: VIF-model is correctly specified;
   - **FLaG-type**: FLaG-IRT is correctly specified;
   - **Bifactor**: testlet nuisance factors, so both models are misspecified.
2. Fits VIF-model along its own L1 (γ) path and selects the graph by BIC, following the procedure of Chen et al. (2018).
3. Compares the BIC-selected solution with a fit given the true graph, pair by pair, by splitting each item-to-item effect into its θ path and S path.

The figures plot the change in the θ path against the change in the S path for every item pair. Points on the slope −1 line mean the total effect stayed the same and only its allocation between *A* and *S* moved.

## How to run

```bash
pip install torch numpy pandas matplotlib joblib
```

- Python 3.10+, CPU only.
- **No data files are needed.** All data are simulated inside the notebook.
- Set `SMOKE = True` in the first code cell to check that everything runs end to end, then use `SMOKE = False` for the full experiment.
- Runtime depends on the number of cores. Set the number of parallel workers with the `IRT_WORKERS` environment variable.
- The notebook is saved without outputs. Run it to reproduce the tables and figures.

## Related

- [FLaG-PISA-Experiment](https://github.com/Hyuckchan-Cha/FLaG-PISA-Experiment): FLaG-IRT and VIF-model on PISA 2018 science data.

## References

The full reference list is in the first cell of the notebook. Key references:

- Chen, Y., Li, X., Liu, J., & Ying, Z. (2018). Robust measurement via a fused latent and graphical item response theory model. *Psychometrika, 83*(3), 538–562.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
- Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory item factor analysis. *Psychometrika, 86*(1), 1–29.

## Author & Contact

Hyuckchan Cha · Seoul National University · [hyuckchan-cha.github.io](https://hyuckchan-cha.github.io)

Questions and comments are welcome: chahchan10039@snu.ac.kr
