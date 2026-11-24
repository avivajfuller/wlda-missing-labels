# =============================================================================
# Missing Labels and WLDA
#
# Note: the code for WLDA is based on python code by the authors found in the github:
# https://github.com/DangLeUyen/WLDA
# Code was made with the help of Claude AI
# Algorithm:
#   1. Remove rows with missing labels temporarily
#   2. Estimate per-class accuracy p_j using 10 repeated 50/50 splits
#      on the fully labeled training data, using WLDA (features may be missing)
#   3. Use WLDA trained on labeled data to pseudo-label unlabeled rows
#   4. Compute adjusted priors:
#        pi_tilde_j = (n_j_true/n) + (n_j_pseudo/n) * p_j
#      then renormalise to sum to 1
#   5. Refit WLDA on all training data (labeled + pseudo-labeled) using
#      the adjusted priors
#   6. Predict test set using WLDA
#
# Two scenarios (matching the paper):
#   Scenario 1: missing labels only (features always complete)
#   Scenario 2: missing labels AND missing features in training + test
#
# Comparison: complete-case WLDA (drops rows with missing labels, keeps
#             WLDA for missing features in remaining rows)
#
# Datasets: Iris, Thyroid, User
# Missing rates: 15%, 30%, 45%, 60%, 75% (matching the paper)
# Each experiment repeated 100 times
# =============================================================================

library(MASS)


# =============================================================================
# SECTION 1: DPER
# Translated from dper.py
# =============================================================================

solving <- function(a, b, c, d, del_case) {
  roots      <- polyroot(c(d, c, b, a))
  real_roots <- Re(roots[abs(Im(roots)) < 1e-8])
  if (length(real_roots) == 0) return(0)
  if (length(real_roots) == 1) return(real_roots[1])
  real_roots[which.min(abs(real_roots - del_case))]
}

sigma_m <- function(X2, y) {
  classes <- sort(unique(y))
  m <- n <- l <- s11 <- s12 <- s22 <- sig11 <- sig22 <- 0
  mus <- lapply(classes, function(g) {
    colMeans(X2[y == g, , drop = FALSE], na.rm = TRUE)
  })
  for (gi in seq_along(classes)) {
    g <- classes[gi]; Xg <- X2[y == g, , drop = FALSE]
    mu0 <- mus[[gi]][1]; mu1 <- mus[[gi]][2]
    for (i in seq_len(nrow(Xg))) {
      x0 <- Xg[i, 1]; x1 <- Xg[i, 2]
      both  <- !is.na(x0) && !is.na(x1)
      only0 <- !is.na(x0) &&  is.na(x1)
      only1 <-  is.na(x0) && !is.na(x1)
      if (both) {
        m <- m+1; s11 <- s11+(x0-mu0)^2; s22 <- s22+(x1-mu1)^2
        s12 <- s12+(x0-mu0)*(x1-mu1)
        sig11 <- sig11+(x0-mu0)^2; sig22 <- sig22+(x1-mu1)^2
      } else if (only0) { n <- n+1; sig11 <- sig11+(x0-mu0)^2
      } else if (only1) { l <- l+1; sig22 <- sig22+(x1-mu1)^2 }
    }
  }
  sig11    <- if ((m+n) > 0) sig11/(m+n) else 0
  sig22    <- if ((m+l) > 0) sig22/(m+l) else 0
  del_case <- if (m > 1) s12/(m-1) else 0
  sig12    <- solving(-m, s12,
                      m*sig11*sig22 - s22*sig11 - s11*sig22,
                      s12*sig11*sig22, del_case)
  c(sig11, sig22, sig12)
}

DPERm <- function(X, y) {
  p <- ncol(X); Sig <- matrix(0, p, p)
  for (a in seq_len(p)) for (b in seq_len(a-1)) {
    tmp <- sigma_m(X[, c(b,a), drop=FALSE], y)
    Sig[b,b] <- tmp[1]; Sig[a,a] <- tmp[2]; Sig[b,a] <- Sig[a,b] <- tmp[3]
  }
  for (i in seq_len(p)) if (Sig[i,i] == 0) {
    obs <- X[!is.na(X[,i]), i]
    Sig[i,i] <- if (length(obs) > 1) var(obs) else 0
  }
  Sig + diag(1e-6, p)
}


# =============================================================================
# SECTION 2: WLDA TRAIN AND PREDICT
# Translated from wlda.py — handles missing features via weight matrix
# =============================================================================

wlda_train <- function(X_train, y_train, prior = NULL) {
  X_train <- as.matrix(X_train)
  classes <- as.character(sort(unique(y_train)))
  n <- nrow(X_train); p <- ncol(X_train)

  # Use supplied priors if given, otherwise compute from class frequencies
  if (is.null(prior)) {
    priors <- sapply(classes, function(g) sum(y_train == as.integer(g)) / n)
  } else {
    priors <- prior[classes]
  }

  mus <- lapply(classes, function(g) {
    colMeans(X_train[y_train == as.integer(g), , drop=FALSE], na.rm=TRUE)
  })
  names(mus) <- classes

  cov_mat <- DPERm(X_train, y_train)

  list(classes              = classes,
       priors               = priors,
       mus                  = mus,
       cov_inv              = solve(cov_mat),
       missing_rate_feature = colMeans(is.na(X_train)),
       p                    = p)
}

wlda_predict <- function(model, X_test) {
  X_test <- as.matrix(X_test)
  w      <- 1 / pmax(1 - model$missing_rate_feature, 1e-4)
  preds  <- character(nrow(X_test))

  for (i in seq_len(nrow(X_test))) {
    x      <- X_test[i, ]
    mask   <- as.numeric(!is.na(x))
    W      <- diag(mask * w, model$p)
    P      <- W %*% model$cov_inv %*% W
    x_fill <- ifelse(is.na(x), 0, x)

    scores <- sapply(model$classes, function(g) {
      diff <- x_fill - model$mus[[g]]
      as.numeric(log(model$priors[g])) -
        0.5 * as.numeric(t(diff) %*% P %*% diff)
    })
    best <- names(which.max(scores))
    if (length(best) == 0) best <- names(which.max(model$priors))
    preds[i] <- best
  }
  preds
}


# =============================================================================
# SECTION 3: SIMULATE MISSINGNESS
# =============================================================================

# Missing labels: randomly set label to NA, keeping at least 2 per class
generate_missing_labels <- function(y, missing_rate) {
  classes <- unique(y); y_miss <- y
  eligible <- c()
  for (g in classes) {
    idx_g <- which(y == g)
    if (length(idx_g) > 2) eligible <- c(eligible, idx_g[3:length(idx_g)])
  }
  n_missing <- min(round(missing_rate * length(y)), length(eligible))
  if (n_missing > 0) y_miss[sample(eligible, n_missing)] <- NA
  y_miss
}

# Missing features: matches generate_NaN from funcs.py
# Keeps column 1 and row 1 fully observed
generate_NaN <- function(X, missing_rate) {
  X_copy    <- X
  X_missing <- X_copy[2:nrow(X_copy), 2:ncol(X_copy), drop=FALSE]
  n_missing <- round(missing_rate * length(X_missing))
  if (n_missing > 0) X_missing[sample(length(X_missing), n_missing)] <- NA
  X_copy[2:nrow(X_copy), 2:ncol(X_copy)] <- X_missing
  X_copy
}


# =============================================================================
# SECTION 4: ESTIMATE PER-CLASS ACCURACY USING WLDA
#
# Same as in WPLDA but uses WLDA instead of LDA for the 50/50 splits.
# Features in the labeled data may already contain NAs (Scenario 2).
# =============================================================================

estimate_class_accuracy_wlda <- function(X_labeled, y_labeled,
                                         n_splits = 10) {
  classes <- sort(unique(y_labeled)); G <- length(classes)

  p_per_split <- matrix(0, nrow = n_splits, ncol = G,
                        dimnames = list(NULL, as.character(classes)))

  for (s in seq_len(n_splits)) {
    n   <- nrow(X_labeled)
    idx <- sample(n, floor(n / 2))

    X_tr <- X_labeled[idx, , drop=FALSE]; y_tr <- y_labeled[idx]
    X_te <- X_labeled[-idx, , drop=FALSE]
    y_te <- as.character(y_labeled[-idx])

    if (length(unique(y_tr)) < G) next

    model <- tryCatch(wlda_train(X_tr, y_tr), error = function(e) NULL)
    if (is.null(model)) next

    preds <- wlda_predict(model, X_te)

    for (g in as.character(classes)) {
      idx_g <- which(y_te == g)
      if (length(idx_g) > 0)
        p_per_split[s, g] <- sum(preds[idx_g] == g) / length(idx_g)
    }
  }

  # Average p_j across splits — if a class never appeared, trust it fully
  p <- colMeans(p_per_split)
  p[p == 0] <- 1

  p
}


# =============================================================================
# SECTION 5: WPWLDA — WEIGHTED PRIOR WLDA
#
# Identical algorithm to WPLDA but uses WLDA throughout.
# Accepts optional X_test with missing features (Scenario 2).
# =============================================================================

wpwlda_fit_predict <- function(X_train, y_train_missing, X_test,
                               n_splits = 10) {
  labeled_idx   <- which(!is.na(y_train_missing))
  unlabeled_idx <- which( is.na(y_train_missing))

  X_labeled <- X_train[labeled_idx, , drop=FALSE]
  y_labeled <- y_train_missing[labeled_idx]

  classes <- sort(unique(y_labeled)); G <- length(classes)
  n       <- nrow(X_train)

  # Step 1: estimate per-class accuracy using WLDA on labeled data
  p_j <- estimate_class_accuracy_wlda(X_labeled, y_labeled, n_splits)

  # Step 2: pseudo-label unlabeled rows using WLDA trained on labeled data
  wlda_base <- wlda_train(X_labeled, y_labeled)

  pseudo_labels <- character(0)
  if (length(unlabeled_idx) > 0) {
    X_unlabeled   <- X_train[unlabeled_idx, , drop=FALSE]
    pseudo_labels <- wlda_predict(wlda_base, X_unlabeled)
  }

  # Combine true and pseudo labels
  y_full <- as.character(y_train_missing)
  if (length(unlabeled_idx) > 0) y_full[unlabeled_idx] <- pseudo_labels

  # Step 3: compute adjusted priors
  pi_tilde <- setNames(numeric(G), as.character(classes))
  for (g in as.character(classes)) {
    n_j_true   <- sum(y_labeled == g)
    n_j_pseudo <- if (length(pseudo_labels) > 0) sum(pseudo_labels == g) else 0
    pi_tilde[g] <- (n_j_true / n) + (n_j_pseudo / n) * p_j[g]
  }
  pi_tilde <- pi_tilde / sum(pi_tilde)   # renormalise

  # Step 4: refit WLDA on all data with adjusted priors
  y_full_int <- as.integer(factor(y_full, levels = as.character(classes))) - 1L
  # Map back to original class integer labels
  y_full_orig <- classes[as.integer(factor(y_full,
                                           levels = as.character(classes)))]

  wlda_final <- wlda_train(X_train, y_full_orig, prior = pi_tilde)

  # Step 5: predict test set
  wlda_predict(wlda_final, X_test)
}


# =============================================================================
# SECTION 6: COMPLETE-CASE WLDA (baseline)
#
# Drops rows with missing labels, fits WLDA on remaining rows.
# Handles missing features naturally via WLDA weighting.
# =============================================================================

complete_case_wlda <- function(X_train, y_train_missing, X_test) {
  labeled_idx <- which(!is.na(y_train_missing))
  X_lab <- X_train[labeled_idx, , drop=FALSE]
  y_lab <- y_train_missing[labeled_idx]

  model <- wlda_train(X_lab, y_lab)
  wlda_predict(model, X_test)
}


# =============================================================================
# SECTION 7: UTILITIES
# =============================================================================

accuracy <- function(y_true, y_pred) {
  mean(as.character(y_true) == as.character(y_pred))
}

train_test_split <- function(X, y, test_size = 0.2) {
  n <- nrow(X); test_idx <- sample(n, round(test_size * n))
  list(X_train = X[setdiff(seq_len(n), test_idx), , drop=FALSE],
       X_test  = X[test_idx, , drop=FALSE],
       y_train = y[setdiff(seq_len(n), test_idx)],
       y_test  = y[test_idx])
}


# =============================================================================
# SECTION 8: EXPERIMENT FUNCTIONS
#
# Scenario 1: missing labels only (features always complete)
# Scenario 2: missing labels AND missing features (same rate for both)
# =============================================================================

scenario1 <- function(X, y, missing_rate, run_time = 100, n_splits = 10) {
  acc_wp <- acc_cc <- numeric(run_time)

  for (rt in seq_len(run_time)) {
    set.seed(rt)
    split   <- train_test_split(X, y, test_size = 0.2)
    X_train <- split$X_train; X_test <- split$X_test
    y_train <- split$y_train; y_test <- split$y_test

    # Missing labels only — features stay complete
    y_train_miss <- generate_missing_labels(y_train, missing_rate)

    acc_wp[rt] <- accuracy(as.character(y_test),
                           wpwlda_fit_predict(X_train, y_train_miss,
                                              X_test, n_splits))
    acc_cc[rt] <- accuracy(as.character(y_test),
                           complete_case_wlda(X_train, y_train_miss, X_test))
  }

  list(wpwlda = c(mean = mean(acc_wp), sd = sd(acc_wp)),
       cc     = c(mean = mean(acc_cc), sd = sd(acc_cc)))
}

scenario2 <- function(X, y, missing_rate, run_time = 100, n_splits = 10) {
  acc_wp <- acc_cc <- numeric(run_time)

  for (rt in seq_len(run_time)) {
    set.seed(rt)
    split   <- train_test_split(X, y, test_size = 0.2)
    y_train <- split$y_train; y_test <- split$y_test

    # Missing features injected before splitting labels
    X_train_nan <- generate_NaN(split$X_train, missing_rate)
    X_test_nan  <- generate_NaN(split$X_test,  missing_rate)

    # Missing labels on top of missing features
    y_train_miss <- generate_missing_labels(y_train, missing_rate)

    acc_wp[rt] <- accuracy(as.character(y_test),
                           wpwlda_fit_predict(X_train_nan, y_train_miss,
                                              X_test_nan, n_splits))
    acc_cc[rt] <- accuracy(as.character(y_test),
                           complete_case_wlda(X_train_nan, y_train_miss,
                                              X_test_nan))
  }

  list(wpwlda = c(mean = mean(acc_wp), sd = sd(acc_wp)),
       cc     = c(mean = mean(acc_cc), sd = sd(acc_cc)))
}

run_experiment <- function(X, y,
                           missing_rates = c(0.15, 0.30, 0.45, 0.60, 0.75),
                           run_time = 100, n_splits = 10) {
  cat("Running Scenario 1 (missing labels only)...\n")
  res1 <- list()
  for (mr in missing_rates) {
    res1[[as.character(mr)]] <- scenario1(X, y, mr, run_time, n_splits)
    cat(sprintf("  %.0f%% done\n", mr * 100))
  }

  cat("Running Scenario 2 (missing labels + missing features)...\n")
  res2 <- list()
  for (mr in missing_rates) {
    res2[[as.character(mr)]] <- scenario2(X, y, mr, run_time, n_splits)
    cat(sprintf("  %.0f%% done\n", mr * 100))
  }

  list(scenario1 = res1, scenario2 = res2)
}

print_results <- function(results, dataset_name) {
  fmt <- function(res, label) {
    cat(sprintf("\n%s | %s\n", dataset_name, label))
    cat(sprintf("%-14s  %-26s  %-26s\n",
                "Missing rate", "WPWLDA (mean ± sd)",
                "Complete-case WLDA (mean ± sd)"))
    cat(strrep("-", 70), "\n")
    for (r in names(res)) {
      cat(sprintf("%-14s  %-26s  %-26s\n",
                  paste0(as.numeric(r)*100, "%"),
                  sprintf("%.3f ± %.3f", res[[r]]$wpwlda["mean"],
                          res[[r]]$wpwlda["sd"]),
                  sprintf("%.3f ± %.3f", res[[r]]$cc["mean"],
                          res[[r]]$cc["sd"])))
    }
  }
  fmt(results$scenario1, "Scenario 1: missing labels only")
  fmt(results$scenario2, "Scenario 2: missing labels + missing features")
  cat("\n")
}


# =============================================================================
# SECTION 9: MAIN
# =============================================================================

missing_rates <- c(0.15, 0.30, 0.45, 0.60, 0.75)

# --- Iris ---
data(iris)
X_iris <- as.matrix(iris[, 1:4])
y_iris <- as.integer(iris$Species) - 1L
cat("=== Iris ===\n")
res_iris <- run_experiment(X_iris, y_iris, missing_rates,
                           run_time = 100, n_splits = 10)
print_results(res_iris, "Iris")

# --- Thyroid ---
cat("=== Thyroid ===\n")
thyroid_raw <- read.csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data",
  header = FALSE)
X_thyroid <- as.matrix(thyroid_raw[, 2:6])
y_thyroid <- as.integer(thyroid_raw[, 1]) - 1L
res_thyroid <- run_experiment(X_thyroid, y_thyroid, missing_rates,
                              run_time = 100, n_splits = 10)
print_results(res_thyroid, "Thyroid")



library(readxl)

train_raw <- read_excel("Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls",
                        sheet = "Training_Data")
test_raw  <- read_excel("Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls",
                        sheet = "Test_Data")

test_raw[[6]] <- gsub("Very Low", "very_low", test_raw[[6]])
user_raw <- rbind(train_raw, test_raw)

X_user <- as.matrix(user_raw[, 1:5])
y_user <- as.integer(as.factor(user_raw[[6]])) - 1L

results_user <- run_experiment(X_user, y_user,
                               missing_rates = c(0.10, 0.20, 0.30, 0.40, 0.50),
                               run_time      = 100,
                               n_splits      = 10)
cat("=== User ===\n")
# https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling
print_results(results_user, "User")
