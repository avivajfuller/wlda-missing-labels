# Handling Missing Labels in Linear Discriminant Analysis

**Aviva Fuller** · Rutgers University · Presented May 2026

📄 [Slides (PDF)](slides/handling-missing-data-in-lda.pdf)

This project extends **Weighted Missing Linear Discriminant Analysis (WLDA)** (Vo et al., 2024) to handle training data where some **class labels** are missing, not just features. The method, which I call **WPWLDA (weighted-prior WLDA)**, adjusts the class priors to reflect how much we trust labels the model predicted itself.

## Background

Linear discriminant analysis (LDA) assumes each class is Gaussian with a shared covariance matrix. It estimates class priors π, class means μ, and the covariance Σ from labeled training data.

**WLDA** (Vo et al., 2024) handles **missing features** directly instead of imputing them first:

- μ and Σ are estimated with the DPER algorithm (Direct Parameter Estimation for Randomly missing data; Nguyen et al., 2022).
- A weight matrix zeros out missing features and down-weights features with high missing rates in the discriminant function.
- With no missing data, WLDA reduces to ordinary LDA.

In the paper's experiments, WLDA beat imputation methods such as MICE and KNN imputation in most settings.

## The open question: missing labels

What if some **labels** in the training data are missing? Existing approaches such as self-learning LDA and EM-LDA (Krijthe & Loog, 2014) predict the missing labels and then **fully trust** those predictions when training.

My method still uses the model to predict missing labels, but it **down-weights predicted labels for classes the model tends to get wrong**.

## Method: WPWLDA

Given training data where only some observations are labeled:

1. **Pseudo-label.** Train WLDA on the labeled rows and predict labels for the unlabeled rows.
2. **Estimate per-class accuracy.** Split the labeled data randomly in half, train WLDA on one half and test on the other. Repeat 10 times and average to get $\hat p_j$, the estimated probability of predicting class $j$ correctly.
3. **Adjust the priors.** Count true and predicted labels separately, and down-weight the predicted ones by $\hat p_j$:

   $$\tilde\pi_j = \frac{n_j^{\text{true}} + \hat p_j \, n_j^{\text{predicted}}}{n}$$

   then renormalize so the priors sum to 1.
4. **Refit and predict.** Estimate μ and Σ as in WLDA on all training data (true plus pseudo-labels), use the adjusted priors $\tilde\pi$, and classify the test set.

With no missing labels, this reduces to WLDA, and with no missing data at all, to ordinary LDA.

## Experiments

The baseline is **complete-case WLDA**, which drops every training row with a missing label.

- **Datasets:** Iris, Thyroid (UCI new-thyroid), User Knowledge Modeling (UCI)
- **Scenario 1:** missing labels only
- **Scenario 2:** missing labels *and* missing features (same rate for both)
- **Missing rates:** 15–75% (10–50% for User)
- Each setting is repeated over many random 80/20 train/test splits; tables report mean ± sd accuracy.

### Results

**Iris**

| Missing rate | S1: WPWLDA | S1: Complete-case | S2: WPWLDA | S2: Complete-case |
|---|---|---|---|---|
| 15% | 0.975 ± 0.024 | 0.975 ± 0.024 | 0.964 ± 0.036 | 0.963 ± 0.035 |
| 30% | 0.974 ± 0.025 | 0.972 ± 0.028 | 0.944 ± 0.039 | 0.943 ± 0.039 |
| 45% | 0.972 ± 0.026 | 0.970 ± 0.028 | 0.920 ± 0.051 | 0.910 ± 0.080 |
| 60% | 0.971 ± 0.027 | 0.965 ± 0.033 | 0.696 ± 0.278 | 0.681 ± 0.319 |
| 75% | 0.966 ± 0.033 | 0.957 ± 0.041 | 0.438 ± 0.254 | 0.457 ± 0.283 |

**Thyroid**

| Missing rate | S1: WPWLDA | S1: Complete-case | S2: WPWLDA | S2: Complete-case |
|---|---|---|---|---|
| 15% | 0.914 ± 0.038 | 0.920 ± 0.038 | 0.913 ± 0.041 | 0.917 ± 0.040 |
| 30% | 0.910 ± 0.041 | 0.920 ± 0.038 | 0.903 ± 0.042 | 0.910 ± 0.041 |
| 45% | 0.907 ± 0.040 | 0.920 ± 0.041 | 0.877 ± 0.058 | 0.852 ± 0.154 |
| 60% | 0.901 ± 0.043 | 0.920 ± 0.040 | 0.778 ± 0.170 | 0.701 ± 0.256 |
| 75% | 0.898 ± 0.045 | 0.919 ± 0.037 | 0.601 ± 0.140 | 0.607 ± 0.149 |

**User Knowledge Modeling**

| Missing rate | S1: WPWLDA | S1: Complete-case | S2: WPWLDA | S2: Complete-case |
|---|---|---|---|---|
| 10% | 0.949 ± 0.020 | 0.950 ± 0.021 | 0.885 ± 0.033 | 0.885 ± 0.034 |
| 20% | 0.947 ± 0.021 | 0.949 ± 0.021 | 0.818 ± 0.043 | 0.819 ± 0.044 |
| 30% | 0.945 ± 0.023 | 0.947 ± 0.021 | 0.746 ± 0.041 | 0.747 ± 0.042 |
| 40% | 0.942 ± 0.024 | 0.947 ± 0.023 | 0.673 ± 0.046 | 0.673 ± 0.045 |
| 50% | 0.939 ± 0.025 | 0.944 ± 0.025 | 0.599 ± 0.054 | 0.598 ± 0.052 |

**Takeaways**

- On **Iris**, WPWLDA matches or slightly beats complete-case WLDA, and the gap grows as more labels are missing.
- On **Thyroid** with both missing labels and missing features, WPWLDA is more accurate at 45–60% missingness, and much less variable (sd 0.058 vs 0.154 at 45%).
- With missing labels only, complete-case WLDA is slightly better on Thyroid, and the two methods are essentially tied on User.

### Future directions

- Test WLDA and WPWLDA on more datasets
- Data that is not missing at random
- Improvements to the method
- Statistical properties of the method

## Running the code

Everything is in [`wpwlda.R`](wpwlda.R). It needs R with the `MASS` and `readxl` packages:

```r
install.packages(c("MASS", "readxl"))
```

Then run:

```bash
Rscript wpwlda.R
```

**Data**

- **Iris** is built into R.
- **Thyroid** is downloaded automatically from the UCI repository.
- **User Knowledge Modeling** must be downloaded from [UCI](https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling). Place `Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls` in the same folder as the script.

With 100 repetitions per setting, a full run takes a while. Lower `run_time` in the `run_experiment` calls for a quick test.

## Acknowledgments

- The WLDA and DPER functions are R translations of the authors' Python code at [DangLeUyen/WLDA](https://github.com/DangLeUyen/WLDA).
- The code was written with the help of Claude AI.

## References

- Krijthe, J. H., & Loog, M. (2014). Implicitly constrained semi-supervised linear discriminant analysis. *Proceedings of the 22nd International Conference on Pattern Recognition*, 3762–3767.
- Nguyen, T., Nguyen-Duy, K. M., Nguyen, D. H. M., Nguyen, B. T., & Wade, B. A. (2022). DPER: Direct parameter estimation for randomly missing data. *Knowledge-Based Systems*, 240, 108082.
- Vo, T. L., Dang, U., & Nguyen, T. (2024). Directly handling missing data in linear discriminant analysis for enhancing classification accuracy and interpretability. *arXiv preprint* arXiv:2407.00710.
