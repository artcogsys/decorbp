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

However, this rule induces jumps if the imposed variance is not the same as the original input variance. That is, in each batch we rescale using a different factor, leading to erratic behaviour.

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



---

We find it convenient to write the update again in terms of the diagonal and off-diagonal parts:
$$
R
$$




---
We may also choose to write this as
$$
R \leftarrow R - \alpha \left[ \kappa (I - N) + (1-\kappa) N C \right] R
$$
where $\alpha \kappa = \epsilon$ and $\alpha(1-\kappa) = \epsilon \eta$. 


This nudges in the right direction via a gradient update. Here, however, the variances will only be shifted in the direction of the imposed variance (rather than forced). We could again introduce a rescaling of the importance of off-diagonal vs diagonal.

recommendation: If we use the full instead of batched data then we can use the first variant since the normalizer does not change in each batch. In batched mode we should use the alternative where we only nudge in the direction of a certain variance.

I also added the option to ignore the normalizer, in which case both options coincide.

### Bias

