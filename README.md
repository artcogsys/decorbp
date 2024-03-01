This repository implements the updated decorrelation approach

## NOTES

### Learning rule

We start from
$$
R \leftarrow n \circ (I - \eta C) R
$$
with normalization vector $n$ and correlation matrix $C$. We may also write this as
$$
R \leftarrow N R - \eta N C R
$$
where $N = \textrm{diag}(n)$. Here, the first term normalizes $R$ to the variance constraint and the second term moves $R$ in the direction of the decorrelated normalized $R$.

However, this rule induces jumps if the imposed variance is not the same as the original input variance. That is, in each batch we rescale using a different factor, leading to erratic behaviour. This is exacerbated in deep networks.

We can alternatively use
$
R \leftarrow R - \epsilon \Delta R
$
with 
$$
\Delta R = R - N R + \eta C R = (I - N + \eta N C) R
$$
which nudges $R$ in the direction of the optimal update. Hence,
$$
R \leftarrow R - \epsilon (I - N + \eta N C) R
$$
where $\epsilon > 0$ controls the step size and $\eta > 0$ controls how much we push towards (normalized) decorrelated states.

[This notebook](examples/decorrelation_analysis.ipynb) shows that this indeed works as expected.

If we use the full instead of batched data then we can use the first variant since the normalizer does not change in each batch. In batched mode we should use the alternative where we only nudge in the direction of a certain variance.

I also added the option to ignore the normalizer (```variance=None```), in which case both options coincide.

### Explicitly controlling variance and correlation

The above nudges in the right direction via a gradient update. Here, however, the variances will only be shifted in the direction of the imposed variance (rather than forced). We could again introduce a rescaling of the importance of off-diagonal vs diagonal.

We may also choose to write this as
$$
R \leftarrow R - \alpha \left[ \kappa (I - N) + (1-\kappa) N C \right] R
$$
so we can more explicitly trade off variance normalization and decorrelation.

### Issues

We are still experiencing issues with multiple layers which is related to the variance constraint. See [this notebook](examples/train_analysis.ipynb) (last example when choosing variance=1.0) as well as the experiments folder. May also have to do with MNIST issues (zeros/scaling) and demeaning.

### Bias

Demeaning using bias needs to more testing. I replaced ```sum``` with ```mean```.

### Downsampling

How many samples needed for proper estimation? dimensionality dependent but can be analytic.

### Triangular part

Should check the theory first but still wondering about how we handle full vs triangular part and its tradeoffs. I also want to check against the old (updated) learning rule.
