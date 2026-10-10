# Encoding techniques

In order for BGV and BFV to be useable, the real valued model weights must be encoded as integers before encryption and decoded after decryption. There are many possible techniques for this, some which I explored in my thesis and many that I did not.

## Techniques used in thesis

In my thesis I mainly used Balanced base-B encoding, section 2.10 of my thesis, and simple rounding to the nearest integer.

Balanced base-B encoding encodes a $r \in \mathbb{R}$ as a integer polynomial $$p(x) = \sum^{n-1}_{i=0} b_i x^i + \sum^m_{j=1} b_{-j}x^{-j},$$
i.e., a polynomial with $n + m$ coefficients. This means that each real number takes up multiple slots in the ciphertext, e.g., $8$ or $16$ as used in the thesis. While Balanced base-B encoding produces models with good performance, the increased number of ciphertexts required greatly increases runtime and memory/network usage. This makes BGV and BFV much worse than CKKS performance wise, since CKKS achieves similarly performing models with less time and resources.

Another technique tested was simply rounding to the nearest integer, i.e., $x = \lfloor r \rceil$. This gave models with horrible performance, but allowed BGV to be faster and use less memory/network than CKKS. Thus, encoding techniques which only use one integer per real number should be explored further, which is the point of this document.

For more detailed discussion of performance, see sections 5.5 and 6.3 of my thesis.

## Other techniques

### Scaling and rounding

Since simply rounding to the nearest integer is not sufficient, one option is to scale before rounding, i.e., $x = \lfloor r\cdot S \rceil$ for some $S$. The main consideration is then the choice of $S$. If it is too small, then performance still suffers, but it must still be small enough to satisfy $C\cdot x \in (-t/2, t/2] \subset \mathbb{Z}$, for $C$ participating clients (All computation must remain in the plaintext space $R_t$). This achieved by using $t = 2\cdot C \cdot X + 1$, where $|x| \leq X$, as in the thesis. If the original real $r$ is bounded $|r| \leq Y$, one gets $t = 2 C S Y + 1$.

Since $S$ affects $t$ it also affects the number of clients that can be used. Recall from section 5.5.4 of my thesis that $C < \frac{q_0}{t}$ for BGV and BFV, meaning that $C < \frac{q_0}{2 C S Y + 1}$. 

Solving for $S$ gives $$S < \frac{q_0 - C}{2C^2Y}$$

In OpenFHE specifically $t$ must satisfy $t < 2^{32}$ (Cannot find proper source for this other than the fact that programs crash when using t = 2^{32} or larger). Thus, $2 C S Y + 1 < 2^{32}$, giving $S < \frac{2^{32} - 1}{2CY}$ as well. Since $q_0$ can be a $60$ bits prime for all parameters in the thesis, $q \geq 2^{59}$ and therefore $$S < \frac{2^{32} - 1}{2CY}$$ is (very likely) the smaller, more relevant bound.

As an example, for SqueezeNet 1.1 with MNIST on $C = 3$ clients, have $Y = 12000$ which gives $S < \frac{2^{32} - 1}{2\cdot 3 \cdot 12000} \approx 59652.32$, so $S$ can be at most $59652$.

### Normalizing, scaling and rounding

Scaling and rounding is promising, but $Y$ can be quite large, which limits the size of $S$. Therefore, one can normalize $r$ to $[0,1]$ before scaling and rounding, i.e., compute $$x = \lfloor S\cdot \frac{r + Y}{2Y} \rceil \in [0, S].$$

This means that $t = 2CS + 1$, which bounds $S$ as
$$S < \frac{2^{32} - 1}{2C}.$$

When $C = 3$, have $S < \frac{2^{32} - 1}{6} = 715827882.5$, $S \leq 715827882$.

Decoding can then be done by computing $$r = \frac{2Yx}{S} - Y.$$