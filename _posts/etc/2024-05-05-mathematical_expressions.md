---
title: "LaTeX Math Command Guide"
categories:
  - ETC
tags:
  - cheatsheet

last_modified_at: 2024-05-05
redirect_from:
    - /ETC/LaTex Math Command Guide
---


# Mathematical Expressions

내가 보려고 만든 LaTeX Math commands 

## Greek letters
        
- Lower case

  |    |           |    |          |    |             |
  |----|-----------|----|----------|----|-------------|
  | α  | \alpha    | β | \beta    | 𝛾 | \gamma      |
  | δ | \delta    | ϵ | \epsilon | ε | \varepsilon |
  | ζ | \zeta     | η | \eta     | θ | \theta      |
  | ϑ | \vartheta | 𝜄ι | \iota    | 𝜅 | \kappa      |
  | λ | \lambda   | μ | \mu      | ν | \nu         |
  | ξ | \xi       | π | \pi      | 𝜛 | \varpi      |
  | ρ | \rho      | ϱ | \varrho  | 𝜎 | \sigma      |
  | τ | \tau      | υ | \upsilon | 𝜙 | \phi        |
  | φ | \varphi   | χ | \chi     | 𝜓 | \psi        |
  | ω | \omega    |    |          |    |           |

- Upper case 

  |    |         |    |          |    |        |
  |----|---------|----|----------|----|--------|
  | Γ  | \Gamma  | Δ | \Delta   | Θ | \Theta |
  | Λ | \Lambda | Ξ | \Xi      | Π | \Pi    |
  | Σ | \Sigma  | Υ | \Upsilon | Φ | \Phi   |
  | Ψ | \Psi    | Ω | \Omega   |    |        |

---

## Basic expression

`$` 하나면 in-line
`$$` 두개면 display

|  |   |  |
| --- | --- | --- |
| Super/sub script | X^{2}_{a}  | $X^{2}_{a}$ |
| Fraction | \frac{dx}{dt} | $\frac{dx}{dt}$ |
| Integral | \int^{b}_{a}{f(x)dx} | $\int^{b}_{a}{f(x)dx}$ |
|Partial derivative|\partial|$\partial$|
| Bracket | \left( a \right) |  $\left( a \right)$ |
| Root | \sqrt[3]{A} | $\sqrt[3]{A}$ |
| Limit | \lim_{a \to 0} | $\lim_{a\to0}$ |
| Infinity | \infty |  $\infty$ |
| Sum | \sum^{n}_{i=0} |  $\sum^{n}_{i+1}$ |
| Prod | \prod^{n}_{i=0} |  $\prod^{n}_{i+1}$ |
|vertical line|\vert|$\vert$|
||\rvert a \lvert|$\rvert a \lvert$|
||\rVert a \lVert|$\rVert a \lVert$|
|Times(multiplication)|\times|$\times$|
| Arrow | \to |  $\to$ |
|1em space|\quad|$a \quad b$|


Tip.  
For multiple integrals, use: \iint $\iint$ \iiint $\iiint$ etc.  
For a closed path integral, use: $\oint$



---

### Dots

||||
|---------------------|-----|--------|
| Multiplication dot  | \cdot  | $\cdot$  |
| Three centered dots | \cdots  | $\cdots$ |
| Three baseline dots | \ldots | $\ldots$ |
| Three diagonal dots | \ddots  | $\ddots$ |
| Three vertical dots | \vdots | $\vdots$ |


---

### Accents

|||||||
|:------:|-----------|--------|----------|------------|---------------|
| $\hat{a}$ | \hat{a}   | $\bar{a}$ | \bar{a}  | $\mathring{a}$  | \mathring{a}  |
| $\check{a}$ | \check{a} | $\dot{a}$ | \dot{a}  | $\vec{a} $ | \vec{a}       |
| $\tilde{a}$ | \tilde{a} | $\ddot{a}$ | \ddot{a} | $\widehat{AAA}$ | \widehat{AAA} |


---

### Relations

|||||||
|------|---------|----|-----------|----|---------|
| $\ne$ | \ne     | $\le$ | \le       | $\ge$ | \ge  |
| $\equiv$  | \equiv  | $\sim $  | \sim      | $\simeq $ | \simeq  |
| $ \approx$   | \approx | $\cong $  | \cong     | $\propto$  | \propto |
| $\mid$  | \mid    | $\parallel$ | \parallel | $\perp$ | \perp  |

---

### Matrix

```tex
 \begin{bmatrix} 
  a & b & c
\\d & e & f
\\g & h & i
\end{bmatrix}
```

$$
\begin{bmatrix}
a & b & c
\\d & e & f
\\g & h & i
\end{bmatrix}
$$

Tip. Use \\\ to separate different rows, and & to separate elements of each row.

- smallmatrix: for inline
- matrix: No delimiter
- pmatrix: ( delimiter
- bmatrix: [ delimiter
- Bmatrix: { delimiter
- vmatrix: $\vert$ delimiter
- Vmatrix: $\Vert$ delimiter

---

### Cases

```tex
\begin{cases}
  x & \text{if } x > 0 \\
  0 & \text{if } x \le 0
\end{cases}
```

$$
\begin{cases}
  x & \text{if } x > 0 \\
  0 & \text{if } x \le 0
\end{cases}
$$

---


### ETC

||||
|---|---|---|
|For all|\forall|$\forall$|
|There exist|\exists|$\exist$|
|Therefore|\therefore|$\therefore$|
|Because|\because|$\because$|
|if a then b (implies) |\implies|$\implies$|
|if and only if|\iif|$\iff$|


