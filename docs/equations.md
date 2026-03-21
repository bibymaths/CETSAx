# Scientific Model

This section formalizes the mathematical framework underlying CETSAx–NADPH. The model integrates dose–response fitting,
sensitivity scoring, and systems-level inference into a unified quantitative pipeline.

---

## 1. Dose–Response Model

Each protein \( i \) is observed across a set of concentrations \( c \in \mathbb{R}_{>0} \). The observed CETSA signal
is denoted:

\[
y_i(c)
\]

### 1.1 Logistic ITDR Model

The response is modeled using a 4-parameter logistic function:

:contentReference[oaicite:0]{index=0}

where:

- \( E_0 \): baseline stability
- \( E_{\max} \): maximal stability shift
- \( EC_{50} \): half-maximal effective concentration
- \( h \): Hill coefficient (cooperativity)

### 1.2 Log-Dose Parameterization

Define:

\[
x = \log_{10}(c)
\]

Then the model becomes:

\[
y(x) = E_0 + \frac{E_{\max} - E_0}{1 + 10^{(\log_{10}(EC_{50}) - x)\cdot h}}
\]

This parameterization improves numerical stability over wide concentration ranges.

---

## 2. Parameter Estimation

Let \( \{(c_j, y_j)\}_{j=1}^n \) denote observed data for a protein.

### 2.1 Weighted Nonlinear Least Squares

Parameters \( \theta = (E_0, E_{\max}, \log EC_{50}, h) \) are estimated by minimizing:

\[
\min_{\theta} \sum_{j=1}^{n} \frac{(y_j - \hat{y}(c_j; \theta))^2}{\sigma_j^2}
\]

where weights are defined as:

\[
\sigma_j = c_j^{\alpha}, \quad \alpha > 0
\]

This emphasizes low-concentration behavior.

---

### 2.2 Regularization

To avoid unrealistic cooperativity:

\[
\mathcal{L}_{\text{reg}} = \lambda (h - h_0)^2
\]

where:

- \( h_0 = 1 \) (target Hill slope)
- \( \lambda > 0 \) controls regularization strength

Total loss:

\[
\mathcal{L} = \sum_{j=1}^{n} \frac{(y_j - \hat{y}(c_j))^2}{\sigma_j^2} + \lambda (h - h_0)^2
\]

---

### 2.3 Monotonicity Constraint

Let \( \tilde{y}_j \) be the isotonic regression estimate of \( y_j \). Then fitting is performed on:

\[
\tilde{y}_j = \text{IsoReg}(y_j)
\]

ensuring monotonicity consistent with stabilization or destabilization.

---

## 3. Fit Diagnostics

### 3.1 Coefficient of Determination

\[
R^2 = 1 - \frac{\sum_j (y_j - \hat{y}_j)^2}{\sum_j (y_j - \bar{y})^2}
\]

### 3.2 Effect Size

\[
\Delta_{\max} = \max_j y_j - \min_j y_j
\]

Only fits satisfying:

- \( R^2 > \tau_R \)
- \( \Delta_{\max} > \tau_\Delta \)

are retained.

---

## 4. Sensitivity Scoring

Each protein is mapped to a scalar **NADPH Sensitivity Score (NSS)**.

### 4.1 Feature Vector

Define:

\[
\mathbf{f}_i = \left( EC_{50,i}, \Delta_{\max,i}, h_i, R^2_i \right)
\]

### 4.2 Robust Scaling

Each feature \( f \) is transformed:

\[
f' = \frac{f - \text{median}(f)}{\text{IQR}(f) + \epsilon}
\]

### 4.3 Directional Transformations

- EC50 is inverted:

\[
EC50^* = -\log_{10}(EC_{50})
\]

- Other features remain monotonic with effect strength.

---

### 4.4 Composite Score

\[
\text{NSS}_i = \sum_{k} w_k \cdot \phi(f'_{i,k})
\]

where:

- \( w_k \): feature weights
- \( \phi(\cdot) \): bounded transformation (e.g. sigmoid)

Typical weighting:

\[
(w_{EC50}, w_{\Delta}, w_h, w_{R^2}) \approx (0.45, 0.30, 0.15, 0.10)
\]

---

## 5. Pathway Enrichment

Let \( S_i \) denote NSS for protein \( i \), and \( \mathcal{P} \subseteq \mathcal{V} \) a pathway.

### 5.1 Continuous Enrichment

Test:

\[
H_0: S_i \sim S_j \quad \forall i \in \mathcal{P}, j \notin \mathcal{P}
\]

using Mann–Whitney U:

\[
U = \sum_{i \in \mathcal{P}} \sum_{j \notin \mathcal{P}} \mathbb{I}(S_i > S_j)
\]

---

### 5.2 Over-Representation

Define hit set:

\[
\mathcal{H} = \{i \mid \text{NSS}_i > \tau\}
\]

Test enrichment via Fisher’s exact test on contingency table:

\[
\begin{pmatrix}
|\mathcal{H} \cap \mathcal{P}| & |\mathcal{H} \setminus \mathcal{P}| \\
|\mathcal{P} \setminus \mathcal{H}| & |\mathcal{V} \setminus (\mathcal{H} \cup \mathcal{P})|
\end{pmatrix}
\]

---

## 6. Network Model

### 6.1 Co-Stabilization Matrix

Let \( \mathbf{x}_i \in \mathbb{R}^d \) be the dose-response vector for protein \( i \).

\[
C_{ij} = \text{corr}(\mathbf{x}_i, \mathbf{x}_j)
\]

---

### 6.2 Graph Construction

Define graph \( G = (V, E) \):

\[
E = \{(i,j) \mid |C_{ij}| \geq \tau\}
\]

with edge weight:

\[
w_{ij} = C_{ij}
\]

---

### 6.3 Community Detection

Modules are identified by maximizing modularity:

\[
Q = \frac{1}{2m} \sum_{i,j} \left( w_{ij} - \frac{k_i k_j}{2m} \right)\delta(c_i, c_j)
\]

where:

- \( k_i \): node degree
- \( m \): total edge weight
- \( c_i \): community assignment

---

## 7. Latent Representation

### 7.1 Feature Matrix

Let:

\[
X \in \mathbb{R}^{n \times p}
\]

be the standardized feature matrix.

---

### 7.2 Principal Component Analysis

Compute:

\[
X = U \Sigma V^T
\]

Latent coordinates:

\[
Z = X V
\]

---

### 7.3 Factor Analysis

Assume:

\[
X = Z \Lambda^T + \epsilon
\]

where:

- \( Z \): latent factors
- \( \Lambda \): loadings
- \( \epsilon \sim \mathcal{N}(0, \Psi)\)

---

## 8. Sequence-Based Model

Let protein sequence be:

\[
s = (a_1, a_2, \dots, a_L)
\]

### 8.1 Embedding

Using a pretrained model:

\[
\mathbf{H} = (h_1, \dots, h_L), \quad h_i \in \mathbb{R}^d
\]

---

### 8.2 Attention Pooling

\[
\alpha_i = \frac{\exp(w^T h_i)}{\sum_j \exp(w^T h_j)}, \quad
z = \sum_{i=1}^{L} \alpha_i h_i
\]

---

### 8.3 Prediction

\[
\hat{y} = f_{\theta}(z)
\]

where \( f_{\theta} \) is a neural network.

---

## 9. Explainability

### 9.1 Saliency

\[
S_i = \left\| \frac{\partial \hat{y}}{\partial h_i} \right\|
\]

---

### 9.2 Integrated Gradients

\[
IG_i = (h_i - h_i^{(0)}) \cdot \int_{0}^{1} \frac{\partial \hat{y}(\alpha h_i)}{\partial h_i} d\alpha
\]

---

## 10. Summary

The CETSAx–NADPH framework defines a mapping:

\[
\text{Dose–response} \rightarrow \text{Parameters} \rightarrow \text{Sensitivity} \rightarrow \text{Systems structure}
\rightarrow \text{Sequence determinants}
\]

Each transformation is explicitly defined and interpretable, allowing both statistical inference and mechanistic
insight.