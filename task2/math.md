# Task 2: Math Derivations — Attention Gradients

Let $Q, K, V$ be the query, key, and value matrices.
Let $S = \frac{QK^T}{\sqrt{d_k}}$ be the unscaled-then-scaled scores matrix of shape $(T, T)$.
Let $P = \text{softmax}(S)$ be the row-wise softmax of $S$, shape $(T, T)$.
Let $A = PV$ be the output of the attention head, shape $(T, d_v)$.

## 1. Warmup: Derive $\partial A/\partial V$
Since $A = PV$ is just a matrix multiplication, this one is pretty straightforward.
For individual elements, $A_{ij} = \sum_{k} P_{ik} V_{kj}$.
Taking the derivative with respect to an element $V_{xy}$:
$$ \frac{\partial A_{ij}}{\partial V_{xy}} = P_{ix} \delta_{jy} $$
where $\delta$ is the Kronecker delta.
In matrix calculus notation, the gradient of a scalar loss $L$ with respect to $V$ is given by:
$$ \frac{\partial L}{\partial V} = P^T \frac{\partial L}{\partial A} $$

## 2. Softmax Jacobian
This part took me a while. Let $p = \text{softmax}(s)$, so $p_i = \frac{e^{s_i}}{\sum_k e^{s_k}}$.
We need the Jacobian entries $\frac{\partial p_i}{\partial s_j}$, which splits into two cases:
By the quotient rule:
**Case 1: $i = j$**
$$ \frac{\partial p_i}{\partial s_i} = \frac{e^{s_i} \left( \sum_k e^{s_k} \right) - e^{s_i} \cdot e^{s_i}}{\left( \sum_k e^{s_k} \right)^2} = \frac{e^{s_i}}{\sum_k e^{s_k}} - \left( \frac{e^{s_i}}{\sum_k e^{s_k}} \right)^2 = p_i - p_i^2 = p_i(1 - p_i) $$

**Case 2: $i \neq j$**
$$ \frac{\partial p_i}{\partial s_j} = \frac{0 \cdot \left( \sum_k e^{s_k} \right) - e^{s_i} \cdot e^{s_j}}{\left( \sum_k e^{s_k} \right)^2} = - \frac{e^{s_i}}{\sum_k e^{s_k}} \frac{e^{s_j}}{\sum_k e^{s_k}} = -p_i p_j $$

Combining these two cases using the Kronecker delta ($\delta_{ij} = 1$ if $i=j$, else $0$):
$$ \frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j) $$

## 3. Main Result: Derive $\partial A/\partial Q$
We want to find how a scalar loss $L$ changes with respect to $Q$. By the chain rule:
$$ \frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} \frac{\partial S}{\partial Q} $$

First, find $\frac{\partial S}{\partial Q}$. Since $S = \frac{QK^T}{\sqrt{d_k}}$, we have:
$$ \frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K $$

Next, we need $\frac{\partial L}{\partial S}$. By the chain rule:
$$ \frac{\partial L}{\partial S_{ij}} = \sum_k \frac{\partial L}{\partial P_{ik}} \frac{\partial P_{ik}}{\partial S_{ij}} $$
Since $P$ is computed row-wise, $\frac{\partial P_{ik}}{\partial S_{ij}} = 0$ if the rows differ (i.e., we only care about elements in the same row $i$).
Let $dP_{ik} = \frac{\partial L}{\partial P_{ik}}$. We know from earlier that $dP = \frac{\partial L}{\partial A} V^T$.
Substituting the Softmax Jacobian:
$$ \frac{\partial L}{\partial S_{ij}} = \sum_k dP_{ik} P_{ik}(\delta_{kj} - P_{ij}) $$
$$ \frac{\partial L}{\partial S_{ij}} = dP_{ij} P_{ij} - P_{ij} \sum_k (dP_{ik} P_{ik}) $$
In matrix form, let $dS = \frac{\partial L}{\partial S}$ and $dP = \frac{\partial L}{\partial P}$. Then:
$$ dS = P \odot \left( dP - \text{row\_sum}(dP \odot P) \right) $$
where $\odot$ is element-wise multiplication.

Finally, we substitute $dS$ back into our first equation:
$$ \frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} dS \cdot K $$

## 4. Interpretation
**Why does the gradient vanish when logits are large?**
When the inputs to softmax are large in magnitude, one probability gets pushed to ~1 and the rest to ~0 (softmax saturates).
Looking at the Jacobian formula we derived: $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$.
When $p_i \approx 1$ or $0$, then $p_i(1 - p_i) \approx 0$ and $-p_i p_j \approx 0$ — the entire Jacobian goes to zero. So $dS \approx 0$ and gradients stop flowing to $Q$ and $K$.
This is exactly why we divide $QK^T$ by $\sqrt{d_k}$ — it keeps the logits from blowing up as $d_k$ grows, so softmax stays in a range where gradients are actually useful. Makes the sqrt(d_k) scaling feel a lot less arbitrary once you see it from the gradient perspective.
