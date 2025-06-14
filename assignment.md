# Management Science Final Project (Spring 2025)
**Instructors: Yen-Chun Chou & Howard Hao-Chun Chuang**

Over the course of this semester, you have mastered three core techniques in management science: optimization, simulation, and forecasting. In this final project, you will do:

1. **Optimization** – Formulate and solve a linear-programming-based spline regression model to flexibly forecast your chosen outcome.
2. **Simulation** – Use your selected regression model to generate predictive scenarios and explore system behavior under uncertainty.
3. **Interval Forecasting** – Construct and evaluate prediction intervals around your forecasts to quantify risk and inform decision-making.

The project is organized into three major parts, each building on the previous to integrate these techniques in a cohesive, practical analysis.

---

## Illustrative Example: Two-Knot Spline Regression

Before diving into the full assignment, here is a simple example for a single predictor $x$ with two knots at its 25th and 75th percentiles, $c_{0.25}$ and $c_{0.75}$:

### 1. Compute knots on your training data:

$$
c_{0.25} = \text{25th percentile of } x, \quad c_{0.75} = \text{75th percentile of } x.
$$

### 2. Define hinge features for each observation $i$:

$$
H_{i,0.25} = \max\{0, x_i - c_{0.25} \}, \quad H_{i,0.75} = \max\{0, x_i - c_{0.75} \}.
$$

### 3. Build the spline regression model:

$$
\hat{y}_i = \beta_0 + \beta_1 x_i + \gamma_{0.25} H_{i,0.25} + \gamma_{0.75} H_{i,0.25} + \gamma_{0.75} H_{i,0.75}.
$$

This yields a piecewise-linear fit with slopes:

- $\beta_1$ for $x \leq c_{0.25}$,
- $\beta_1 + \gamma_{0.25}$ for $c_{0.25} < x \leq c_{0.75}$,
- $\beta_1 + \gamma_{0.25} + \gamma_{0.75}$ for $x > c_{0.75}$.

---

## Part I. Unconstrained Spline Regression with 5-Fold CV

### 1. Data Preparation & Splitting

1. Randomly partition the full dataset into five equal-sized folds:  
   $F_1, F_2, F_3, F_4, F_5$

2. For each fold index $k = 1, \dots, 5$:
   - **Training set**: $T_k = \bigcup_{t \neq k} F_t$
   - **Test (validation) fold**: $F_k$

### 2. Knot Selection on Training Data

For each fold $k$ and predictor $x_j \ (j = 1, \dots, p)$, compute on $T_k$:

- $c_{j,0.10} = \text{10th percentile of } x_j$
- $c_{j,0.25} = \text{25th percentile}$
- $c_{j,0.50} = \text{50th percentile (median)}$
- $c_{j,0.75} = \text{75th percentile}$
- $c_{j,0.90} = \text{90th percentile}$

### 3. Feature Construction & Regression Function

1. **Hinge features**: for each observation $i$ and knot $q \in \{0.10, 0.25, 0.50, 0.75, 0.90\}$:

   $$
   H_{i,j,q} = \max\{0, x_{i,j} - c_{j,q} \}
   $$

2. **Full linear predictor**:

   $$
   \hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{i,j} + \sum_{j=1}^{p} \sum_{q \in \{0.10,0.25,0.50,0.75,0.90\}} \gamma_{j,q} H_{i,j,q}
   $$

3. **Model fitting (Gurobi LP)**: on $T_k$, solve

   $$
   \min_{\beta_0, \{\beta_j\}, \{\gamma_{j,q}\}} \sum_{i \in T_k} | y_i - \hat{y}_i |
   $$

   via Gurobi’s LP interface.

### 4. Out-of-Fold Evaluation

1. For each $i \in F_k$, predict $\hat{y}_i$ with fold-$k$ coefficients.
2. Compute fold MAE:

   $$
   \text{MAE}_k = \frac{1}{|F_k|} \sum_{i \in F_k} | y_i - \hat{y}_i |
   $$

3. Report overall CV-MAE:

   $$
   \text{CV-MAE} = \frac{1}{5} \sum_{k=1}^5 \text{MAE}_k
   $$

---

## Part II. $L_1$-Budgeted Spline Regression & Model Comparison

### 1. Define $L_1$ Budget Levels

Choose budgets $B_1 < B_2 < \cdots < B_L$, each enforcing:

$$
\sum_{j=1}^p |\beta_j| + \sum_{j=1}^p \sum_q |\gamma_{j,q}| \leq B_t
$$

### 2. Budgeted CV Grid Search

For each $B_t$:

1. Repeat the Part I 5-fold CV, adding the above $L_1$ constraint to each Gurobi LP on $T_k$.
2. Compute $\text{CV-MAE}(B_t)$

### 3. Model Comparison

1. Select optimal budget:

   $$
   B^* = \arg \min_{B_t} \text{CV-MAE}(B_t)
   $$

2. Compare the unconstrained model vs. the budgeted model at $B^*$ in terms of CV-MAE.
3. Refit the model using optimal $B^*$ and full data. Report and discuss:
   - **Number of nonzero coefficients** ($| \cdot | > 0.001$)
   - **Effect of spline coefficients** (estimated gammas)

---

## Part III. Prediction Interval Construction & Evaluation

### 1. Residual Variance Modeling

For each fold $k$ under budget $B^*$:

1. Compute residuals on $T_k$: $r_i = y_i - \hat{y}_i$
2. Regress $\ln(r_i^2)$ on the same spline features to estimate $\sigma^2(x)$

### 2. Laplace-Based Interval Simulation

For each test point $i \in F_k$:

1. Compute $\hat{y}_i$ and $\hat{\sigma}_i^2$
2. Set $b_i = \sqrt{\hat{\sigma}_i^2}/\sqrt{2}$
3. Draw $S = 100$ errors $e_i^{(s)} \sim \text{Laplace}(0, b_i)$
4. Simulate $y_i^{(s)} = \hat{y}_i + e_i^{(s)}$
5. Interval $[L_i, U_i]$: 5th & 95th percentiles of $\{ y_i^{(s)} \}$

### 3. Evaluation Metrics

- **Coverage rate**:

  $$
  \frac{1}{n} \sum_{i=1}^{n} \mathbf{1} \{ y_i \in [L_i, U_i] \}
  $$

- **MSIS** ($\alpha = 0.10$, fold mean $\bar{y}_k$):

  $$
  \text{MSIS}_i = \frac{U_i - L_i}{\bar{y}_k} + \frac{2}{\alpha} \frac{\max(0, L_i - y_i)}{\bar{y}_k} + \frac{2}{\alpha} \frac{\max(0, y_i - U_i)}{\bar{y}_k}
  $$

---

> **Note**: Report averages across folds.

---

## Bonus Part: Creative Extensions

Optionally, propose and implement a meaningful extension of the methods above that adds value for decision-makers or stakeholders. Briefly describe your idea and why it enhances the base analysis. If you use Generative AI to create extensions, make sure you are able to justify well why such extensions make sense to your data and problem. **Don't throw lots of extensions to us, just get to the point and sharpen your focus.**

---

## Final Assignment (**Important!!!**)

### 1. **Dataset & Implementation**  
Select a real dataset relevant to your domain and perform **all three parts** of the analysis above.

### 2. **Interpretation & Implications**  
For each part, discuss the practical implications for decision-makers or stakeholders, and justify why spline regression (and any chosen extensions) is appropriate and informative for your data and problem context.

### 3. **Deliverable**  
Submit **ALL** following items to Moodle by **23:59 on June 17**:

i) A PowerPoint file with a link to your **presentation video clip** (no more than 12 mins) that clearly explains what you have done and found.  
→ Make sure the link to your video works!

ii) A **clearly and succinctly written report** with necessary equations, tables, and/or graphics.

iii) **Python code and data** (we will check this).

---

## Academic Integrity & AI Usage Policy

Using generative AI tools to produce solutions without true understanding undermines learning. I will easily detect reports filled with AI-generated content that lacks coherence or your own reasoning.  
**If I detect you rely on AI without comprehension, you will be heavily penalized.**  
Demonstrate **genuine engagement and understanding** in your analysis.
