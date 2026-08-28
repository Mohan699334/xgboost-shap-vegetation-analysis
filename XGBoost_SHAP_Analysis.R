# ============================================================
# XGBoost + SHAP analysis for vegetation response to climate extremes
# Public reproducibility script
#
# Workflow:
#   1. Read a user-prepared tabular dataset (or auto-generate demo data)
#   2. Randomly sample up to 10,000 pixels per year
#   3. Construct a 5 x 5 equal-interval spatial grid
#   4. Perform 5-fold spatial block cross-validation
#   5. Tune XGBoost hyperparameters by grid search
#   6. Evaluate the selected model with spatial CV
#   7. Perform temporal validation (<=2017 training; >2017 testing)
#   8. Fit the final model for SHAP interpretation
#   9. Calculate SHAP values and feature importance
#  10. Detect SHAP zero-crossing points from LOESS (span = 0.75)
#  11. Calculate equal-interval 95% confidence intervals for SHAP means
# IMPORTANT:
#   - Raw input data are NOT included in this repository.
#   - If no data file is found, a synthetic dataset with known ecological
#     relationships is automatically generated for testing the code pipeline.
#   - This script uses regression (reg:squarederror).
#   - SHAP values indicate contributions to model predictions; zero-crossing
#     points are model-derived turning points, not confirmed ecological thresholds.
# ============================================================

# -------------------- 0. Packages --------------------
required_packages <- c("xgboost", "ggplot2", "dplyr", "reshape2")
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Please install the following R packages before running the script: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(xgboost)
library(ggplot2)
library(dplyr)
library(reshape2)

# -------------------- 1. User settings --------------------
# Replace INPUT_FILE with your own data file.
INPUT_FILE <- file.path("data", "pixel_samples_XGBoost.csv")
OUTPUT_DIR <- "results"

SEED <- 123
N_PIXELS_PER_YEAR <- 10000
N_SPATIAL_FOLDS <- 5
TRAIN_END_YEAR <- 2017
LOESS_SPAN <- 0.75
N_CI_INTERVALS <- 20
N_SHAP_SAMPLES <- NULL   # NULL = use all final-model samples

# Features and target used in the manuscript analysis
TARGET_COL <- "NDVI"
FEATURE_COLS <- c(
  "CDD", "RX5day", "TXx", "TNn", "DTR",
  "TX10p", "TN10p", "TX90p", "TN90p"
)

set.seed(SEED)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(INPUT_FILE), showWarnings = FALSE, recursive = TRUE)

# -------------------- 2. Auto-generate demo data if real data not found --------------------
if (!file.exists(INPUT_FILE)) {
  cat("\n============================================================\n")
  cat("⚠️  Real data file not found.\n")
  cat("   Generating a synthetic dataset with known ecological relationships...\n")
  cat("   (This is for testing the code pipeline only. Replace with real data for actual analysis.)\n")
  cat("============================================================\n\n")
  
  # 模拟参数：100个像素 × 23年 = 2300个观测
  n_pixels <- 100
  n_years <- 23
  n <- n_pixels * n_years
  
  # 基础空间异质性（每个像素有差异）
  pixel_effects <- rnorm(n_pixels, mean = 0.5, sd = 0.15)
  pixel_effects <- rep(pixel_effects, each = n_years)
  
  # 年际气候变异（每年有波动）
  year_effects <- rep(rnorm(n_years, mean = 0, sd = 0.05), times = n_pixels)
  
  # 生成9个极端气候指数（含空间异质性 + 年际变异）
  # CDD: 连续干旱日数（5~80天）
  CDD_raw <- 20 + 15 * sin(1:n / 500) + rnorm(n, 0, 8)
  CDD <- pmax(3, pmin(80, CDD_raw))
  
  # RX5day: 最大5日降水量（10~150mm）
  RX5day_raw <- 50 + 30 * cos(1:n / 300) + rnorm(n, 0, 15)
  RX5day <- pmax(8, pmin(150, RX5day_raw))
  
  # TXx: 最高温极值（25~45°C）
  TXx_raw <- 32 + 5 * sin(1:n / 400) + rnorm(n, 0, 3)
  TXx <- pmax(22, pmin(46, TXx_raw))
  
  # TNn: 最低温极值（-20~5°C）
  TNn_raw <- -8 + 10 * cos(1:n / 350) + rnorm(n, 0, 4)
  TNn <- pmax(-25, pmin(8, TNn_raw))
  
  # DTR: 昼夜温差（5~20°C）
  DTR_raw <- 12 + 4 * sin(1:n / 250) + rnorm(n, 0, 2)
  DTR <- pmax(3, pmin(22, DTR_raw))
  
  # TX10p: 冷昼日数（0~30天）
  TX10p_raw <- 8 + 5 * cos(1:n / 300) + rnorm(n, 0, 3)
  TX10p <- pmax(0, pmin(35, TX10p_raw))
  
  # TN10p: 冷夜日数（0~40天）
  TN10p_raw <- 12 + 6 * sin(1:n / 280) + rnorm(n, 0, 4)
  TN10p <- pmax(0, pmin(45, TN10p_raw))
  
  # TX90p: 暖昼日数（0~40天）
  TX90p_raw <- 10 + 8 * sin(1:n / 320) + rnorm(n, 0, 4)
  TX90p <- pmax(0, pmin(45, TX90p_raw))
  
  # TN90p: 暖夜日数（0~50天）
  TN90p_raw <- 8 + 10 * sin(1:n / 300) + rnorm(n, 0, 5)
  TN90p <- pmax(0, pmin(55, TN90p_raw))
  
  # ============================================================
  # ★★★ 核心：构建NDVI与气候指数的非线性关系 ★★★
  # ============================================================
  
  # 1) CDD效应：负向饱和（指数衰减型）
  cdd_effect <- -0.15 * (1 - exp(-CDD / 25))
  
  # 2) RX5day效应：正向饱和（Michaelis-Menten型）
  rx5day_effect <- 0.20 * (RX5day) / (RX5day + 40)
  
  # 3) TXx效应：高温抑制（线性负向 + 高温放大）
  txx_effect <- -0.008 * (TXx - 25) * (1 + 0.03 * pmax(0, TXx - 32))
  
  # 4) TNn效应：低温促进（线性正向）
  tnn_effect <- 0.005 * (TNn + 5)
  
  # 5) DTR效应：大温差抑制（二次型）
  dtr_effect <- -0.003 * (DTR - 12)^2
  
  # 6) TX10p/TN10p效应：冷日数促进（弱正效应）
  tx10p_effect <- 0.003 * TX10p
  tn10p_effect <- 0.002 * TN10p
  
  # 7) TX90p/TN90p效应：暖日数抑制（弱负效应）
  tx90p_effect <- -0.004 * TX90p
  tn90p_effect <- -0.003 * TN90p
  
  # NDVI = 基础水平 + 所有效应 + 年际波动 + 随机噪声
  NDVI_raw <- 0.50 +
    cdd_effect +
    rx5day_effect +
    txx_effect +
    tnn_effect +
    dtr_effect +
    tx10p_effect +
    tn10p_effect +
    tx90p_effect +
    tn90p_effect +
    pixel_effects +
    year_effects +
    rnorm(n, 0, 0.04)
  
  NDVI <- pmax(0.05, pmin(0.95, NDVI_raw))
  
  # ============================================================
  # 组装成数据框
  # ============================================================
  
  demo_data <- data.frame(
    year = rep(2000:(2000 + n_years - 1), times = n_pixels),
    x = rep(runif(n_pixels, 70, 140), each = n_years) + rnorm(n, 0, 0.5),
    y = rep(runif(n_pixels, 15, 55), each = n_years) + rnorm(n, 0, 0.5),
    NDVI = NDVI,
    CDD = CDD,
    RX5day = RX5day,
    TXx = TXx,
    TNn = TNn,
    DTR = DTR,
    TX10p = TX10p,
    TN10p = TN10p,
    TX90p = TX90p,
    TN90p = TN90p
  )
  
  # 确保所有值有效
  demo_data <- demo_data[apply(demo_data[, c(FEATURE_COLS, TARGET_COL)], 1, function(row) all(is.finite(row))), ]
  
  # 保存
  write.csv(demo_data, INPUT_FILE, row.names = FALSE)
  
  cat("✅ Synthetic dataset generated with known ecological relationships.\n")
  cat("   Observations:", nrow(demo_data), "\n")
  cat("   NDVI range:", round(min(demo_data$NDVI), 3), "-", round(max(demo_data$NDVI), 3), "\n")
  cat("   Years:", paste(sort(unique(demo_data$year)), collapse = ", "), "\n\n")
  cat("   *** This is DEMO data for code testing only. ***\n")
  cat("   *** Replace with real data for publication results. ***\n")
  cat("============================================================\n\n")
}

# -------------------- 3. Helper functions --------------------
# All helper functions are defined here so the script is self-contained.

# Standardization: use training set statistics, apply to test set
scale_by_train <- function(train_df, test_df, features) {
  train_mean <- sapply(train_df[, features, drop = FALSE], mean, na.rm = TRUE)
  train_sd <- sapply(train_df[, features, drop = FALSE], sd, na.rm = TRUE)
  train_sd[train_sd == 0 | is.na(train_sd)] <- 1
  
  train_scaled <- train_df
  test_scaled <- test_df
  train_scaled[, features] <- scale(train_df[, features, drop = FALSE], center = train_mean, scale = train_sd)
  test_scaled[, features] <- scale(test_df[, features, drop = FALSE], center = train_mean, scale = train_sd)
  
  list(train = train_scaled, test = test_scaled, mean = train_mean, sd = train_sd)
}

# Calculate evaluation metrics (with robust error handling)
calc_metrics <- function(obs, pred) {
  if (length(obs) < 2 || length(pred) < 2 || length(obs) != length(pred)) {
    return(data.frame(R2 = NA_real_, RMSE = NA_real_, MAE = NA_real_))
  }
  valid_idx <- is.finite(obs) & is.finite(pred)
  if (sum(valid_idx) < 2) {
    return(data.frame(R2 = NA_real_, RMSE = NA_real_, MAE = NA_real_))
  }
  obs_clean <- obs[valid_idx]
  pred_clean <- pred[valid_idx]
  data.frame(
    R2 = cor(obs_clean, pred_clean)^2,
    RMSE = sqrt(mean((obs_clean - pred_clean)^2)),
    MAE = mean(abs(obs_clean - pred_clean))
  )
}

# Train XGBoost once with early stopping
train_xgboost_once <- function(train_df, test_df, features, target, params, nrounds = 800, seed = 123) {
  set.seed(seed)
  inner_idx <- sample(1:nrow(train_df), size = floor(0.8 * nrow(train_df)))
  train_inner <- train_df[inner_idx, , drop = FALSE]
  valid_inner <- train_df[-inner_idx, , drop = FALSE]
  
  scaled <- scale_by_train(train_inner, rbind(valid_inner, test_df), features)
  n_valid <- nrow(valid_inner)
  valid_scaled <- scaled$test[1:n_valid, , drop = FALSE]
  test_scaled <- scaled$test[(n_valid + 1):nrow(scaled$test), , drop = FALSE]
  
  dtrain <- xgb.DMatrix(as.matrix(scaled$train[, features, drop = FALSE]), label = scaled$train[[target]])
  dvalid <- xgb.DMatrix(as.matrix(valid_scaled[, features, drop = FALSE]), label = valid_scaled[[target]])
  dtest <- xgb.DMatrix(as.matrix(test_scaled[, features, drop = FALSE]), label = test_scaled[[target]])
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = list(train = dtrain, eval = dvalid),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  pred <- predict(model, dtest)
  metrics <- calc_metrics(test_scaled[[target]], pred)
  
  if (is.null(metrics) || nrow(metrics) == 0) {
    metrics <- data.frame(R2 = NA_real_, RMSE = NA_real_, MAE = NA_real_)
  }
  list(model = model, metrics = metrics, nrounds = ifelse(is.null(model$best_iteration), 0, model$best_iteration))
}

# Run grid search with spatial cross-validation
run_grid_search <- function(data, feature_cols, target_col, param_grid, n_spatial_folds = 5, seed = 123) {
  set.seed(seed)
  unique_blocks <- sort(unique(data$block_id))
  block_fold <- sample(rep(1:n_spatial_folds, length.out = length(unique_blocks)))
  block_to_fold <- setNames(block_fold, unique_blocks)
  data$fold <- unname(block_to_fold[as.character(data$block_id)])
  
  results <- list()
  
  for (g in 1:nrow(param_grid)) {
    params <- list(
      objective = "reg:squarederror",
      max_depth = param_grid$max_depth[g],
      eta = param_grid$eta[g],
      subsample = 0.8,
      colsample_bytree = 0.8,
      min_child_weight = 3,
      gamma = 0.1,
      nthread = 2
    )
    
    fold_R2 <- c()
    for (fold in 1:n_spatial_folds) {
      test_idx <- which(data$fold == fold)
      train_idx <- which(data$fold != fold)
      result <- train_xgboost_once(
        train_df = data[train_idx, , drop = FALSE],
        test_df = data[test_idx, , drop = FALSE],
        features = feature_cols,
        target = target_col,
        params = params,
        nrounds = param_grid$nrounds[g],
        seed = seed + g + fold
      )
      fold_R2 <- c(fold_R2, result$metrics$R2)
    }
    
    results[[g]] <- data.frame(
      grid_id = g,
      max_depth = param_grid$max_depth[g],
      eta = param_grid$eta[g],
      nrounds = param_grid$nrounds[g],
      mean_R2 = mean(fold_R2, na.rm = TRUE),
      sd_R2 = sd(fold_R2, na.rm = TRUE)
    )
  }
  
  do.call(rbind, results)
}

# Run spatial cross-validation with fixed hyperparameters
run_spatial_cv <- function(data, feature_cols, target_col, best_params, best_nrounds, n_spatial_folds = 5, seed = 123) {
  set.seed(seed)
  unique_blocks <- sort(unique(data$block_id))
  block_fold <- sample(rep(1:n_spatial_folds, length.out = length(unique_blocks)))
  block_to_fold <- setNames(block_fold, unique_blocks)
  data$fold <- unname(block_to_fold[as.character(data$block_id)])
  
  results <- list()
  for (fold in 1:n_spatial_folds) {
    test_idx <- which(data$fold == fold)
    train_idx <- which(data$fold != fold)
    result <- train_xgboost_once(
      train_df = data[train_idx, , drop = FALSE],
      test_df = data[test_idx, , drop = FALSE],
      features = feature_cols,
      target = target_col,
      params = best_params,
      nrounds = best_nrounds,
      seed = seed + fold
    )
    results[[fold]] <- result$metrics
  }
  do.call(rbind, results)
}

# Build long-format SHAP data frame (in-memory only, not exported)
build_shap_long <- function(X_original, shap_values, feature_names) {
  shap_df <- as.data.frame(shap_values)
  colnames(shap_df) <- feature_names
  shap_df$sample_id <- 1:nrow(shap_df)
  shap_long <- melt(shap_df, id.vars = "sample_id", variable.name = "feature", value.name = "shap")
  X_long <- melt(cbind(sample_id = 1:nrow(X_original), X_original),
                 id.vars = "sample_id", variable.name = "feature", value.name = "value")
  shap_long <- merge(shap_long, X_long, by = c("sample_id", "feature"))
  shap_long
}

# Calculate equal-interval 95% CI for SHAP means (in-memory only, not exported)
calc_equal_interval_ci <- function(df, x_col, shap_col, n_intervals = 20) {
  x_min <- min(df[[x_col]], na.rm = TRUE)
  x_max <- max(df[[x_col]], na.rm = TRUE)
  if (!is.finite(x_min) || !is.finite(x_max) || x_max <= x_min) {
    return(data.frame())
  }
  breaks <- seq(x_min, x_max, length.out = n_intervals + 1)
  
  results <- list()
  for (i in 1:n_intervals) {
    idx <- df[[x_col]] >= breaks[i] & df[[x_col]] < breaks[i + 1]
    z <- df[[shap_col]][idx]
    z <- z[is.finite(z)]
    if (length(z) >= 2) {
      mean_shap <- mean(z)
      se_shap <- sd(z) / sqrt(length(z))
      results[[i]] <- data.frame(
        x_min = breaks[i],
        x_max = breaks[i + 1],
        n = length(z),
        mean_shap = mean_shap,
        CI_lower = mean_shap - 1.96 * se_shap,
        CI_upper = mean_shap + 1.96 * se_shap
      )
    }
  }
  do.call(rbind, results)
}

# Find SHAP zero-crossing points using LOESS (in-memory only, not exported)
find_thresholds <- function(x, y, span = 0.75) {
  tmp <- data.frame(x = x, y = y)
  tmp <- tmp[is.finite(tmp$x) & is.finite(tmp$y), ]
  if (nrow(tmp) <= 20) return(numeric(0))
  
  loess_fit <- loess(y ~ x, data = tmp, span = span, control = loess.control(surface = "direct"))
  x_seq <- seq(min(tmp$x), max(tmp$x), length.out = 1000)
  y_pred <- predict(loess_fit, newdata = data.frame(x = x_seq))
  
  changes <- which(diff(sign(y_pred)) != 0)
  if (length(changes) == 0) return(numeric(0))
  
  thresholds <- sapply(changes, function(i) {
    x1 <- x_seq[i]; x2 <- x_seq[i + 1]
    y1 <- y_pred[i]; y2 <- y_pred[i + 1]
    x1 - y1 * (x2 - x1) / (y2 - y1)
  })
  unique(round(thresholds, 4))
}

# -------------------- 4. Read and validate data --------------------
cat("\n============================================================\n")
cat("1. DATA PREPARATION\n")
cat("============================================================\n")

data_raw <- read.csv(INPUT_FILE, header = TRUE, check.names = FALSE)

required_cols <- c("year", "x", "y", TARGET_COL, FEATURE_COLS)
missing_cols <- setdiff(required_cols, names(data_raw))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

data <- data_raw[, required_cols, drop = FALSE]

numeric_cols <- c("year", "x", "y", TARGET_COL, FEATURE_COLS)
for (nm in numeric_cols) {
  data[[nm]] <- suppressWarnings(as.numeric(data[[nm]]))
}

valid_rows <- apply(data[, c(FEATURE_COLS, TARGET_COL), drop = FALSE], 1, function(row) {
  all(is.finite(row))
})
valid_rows <- valid_rows & is.finite(data$year) & is.finite(data$x) & is.finite(data$y)
data <- data[valid_rows, , drop = FALSE]

if (nrow(data) == 0) stop("No valid observations remain after removing NA/NaN/Inf values.")

cat("Valid rows before annual sampling:", nrow(data), "\n")

# -------------------- 5. Annual random sampling --------------------
cat("\n============================================================\n")
cat("2. ANNUAL SAMPLING\n")
cat("============================================================\n")

set.seed(SEED)
data <- data %>%
  group_by(year) %>%
  group_modify(~ {
    n_take <- min(N_PIXELS_PER_YEAR, nrow(.x))
    dplyr::slice_sample(.x, n = n_take, replace = FALSE)
  }) %>%
  ungroup()

cat("Rows after annual sampling:", nrow(data), "\n")
cat("Years:", paste(sort(unique(data$year)), collapse = ", "), "\n")

# -------------------- 6. 5 x 5 equal-interval spatial grid --------------------
cat("\n============================================================\n")
cat("3. SPATIAL BLOCKS\n")
cat("============================================================\n")

if (length(unique(data$x)) < 2 || length(unique(data$y)) < 2) {
  stop("Spatial coordinates do not span a 2D area; a 5 x 5 spatial grid cannot be constructed.")
}

x_breaks <- seq(min(data$x), max(data$x), length.out = 6)
y_breaks <- seq(min(data$y), max(data$y), length.out = 6)

data$x_block <- cut(data$x, breaks = x_breaks, labels = FALSE, include.lowest = TRUE)
data$y_block <- cut(data$y, breaks = y_breaks, labels = FALSE, include.lowest = TRUE)
data$block_id <- (data$y_block - 1) * 5 + data$x_block

cat("Number of spatial blocks represented in the data:", length(unique(data$block_id)), "\n")

# -------------------- 7. Assign spatial blocks to 5 folds --------------------
set.seed(SEED)
unique_blocks <- sort(unique(data$block_id))
if (length(unique_blocks) < N_SPATIAL_FOLDS) {
  stop("The number of occupied spatial blocks is smaller than the requested number of folds.")
}

block_fold <- sample(rep(seq_len(N_SPATIAL_FOLDS), length.out = length(unique_blocks)))
block_to_fold <- setNames(block_fold, unique_blocks)
data$fold <- unname(block_to_fold[as.character(data$block_id)])

# -------------------- 8. Grid search --------------------
cat("\n============================================================\n")
cat("4. GRID SEARCH + SPATIAL CV\n")
cat("============================================================\n")

param_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  nrounds = c(100, 200)
)

grid_results <- run_grid_search(
  data = data,
  feature_cols = FEATURE_COLS,
  target_col = TARGET_COL,
  param_grid = param_grid,
  n_spatial_folds = N_SPATIAL_FOLDS,
  seed = SEED
)

best_idx <- which.max(grid_results$mean_R2)
best_params <- list(
  objective = "reg:squarederror",
  max_depth = grid_results$max_depth[best_idx],
  eta = grid_results$eta[best_idx],
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  gamma = 0.1,
  nthread = 2
)
best_nrounds <- grid_results$nrounds[best_idx]

cat("\nSelected hyperparameters:\n")
print(best_params)
cat("Selected nrounds:", best_nrounds, "\n")

# -------------------- 9. Final 5-fold spatial CV --------------------
cat("\n============================================================\n")
cat("5. FINAL SPATIAL CV\n")
cat("============================================================\n")

cv_df <- run_spatial_cv(
  data = data,
  feature_cols = FEATURE_COLS,
  target_col = TARGET_COL,
  best_params = best_params,
  best_nrounds = best_nrounds,
  n_spatial_folds = N_SPATIAL_FOLDS,
  seed = SEED
)

cv_mean <- colMeans(cv_df[, c("R2", "RMSE", "MAE")], na.rm = TRUE)
cv_sd <- apply(cv_df[, c("R2", "RMSE", "MAE")], 2, sd, na.rm = TRUE)

cat(sprintf("Spatial CV R2   = %.4f +/- %.4f\n", cv_mean["R2"], cv_sd["R2"]))
cat(sprintf("Spatial CV RMSE = %.4f +/- %.4f\n", cv_mean["RMSE"], cv_sd["RMSE"]))
cat(sprintf("Spatial CV MAE  = %.4f +/- %.4f\n", cv_mean["MAE"], cv_sd["MAE"]))

# -------------------- 10. Temporal validation --------------------
cat("\n============================================================\n")
cat("6. TEMPORAL VALIDATION\n")
cat("============================================================\n")

train_time <- data[data$year <= TRAIN_END_YEAR, , drop = FALSE]
test_time <- data[data$year > TRAIN_END_YEAR, , drop = FALSE]

if (nrow(train_time) == 0 || nrow(test_time) == 0) {
  cat("⚠️ Temporal validation skipped: one of the two periods has no observations.\n")
} else {
  cat("Training observations (<=", TRAIN_END_YEAR, "):", nrow(train_time), "\n")
  cat("Testing observations (>", TRAIN_END_YEAR, "):", nrow(test_time), "\n")
  
  time_result <- train_xgboost_once(
    train_df = train_time,
    test_df = test_time,
    features = FEATURE_COLS,
    target = TARGET_COL,
    params = best_params,
    nrounds = best_nrounds,
    seed = SEED
  )
  
  cat(sprintf("Temporal validation: R2=%.4f, RMSE=%.4f, MAE=%.4f\n",
              time_result$metrics$R2,
              time_result$metrics$RMSE,
              time_result$metrics$MAE))
}

# -------------------- 11. Final model for SHAP --------------------
cat("\n============================================================\n")
cat("7. FINAL MODEL\n")
cat("============================================================\n")

set.seed(SEED)
inner_idx <- sample(seq_len(nrow(data)), size = floor(0.8 * nrow(data)))
train_final <- data[inner_idx, , drop = FALSE]
valid_final <- data[-inner_idx, , drop = FALSE]

scaled_final <- scale_by_train(train_final, valid_final, FEATURE_COLS)

dtrain_final <- xgb.DMatrix(
  as.matrix(scaled_final$train[, FEATURE_COLS, drop = FALSE]),
  label = scaled_final$train[[TARGET_COL]]
)
dvalid_final <- xgb.DMatrix(
  as.matrix(scaled_final$test[, FEATURE_COLS, drop = FALSE]),
  label = scaled_final$test[[TARGET_COL]]
)

final_model <- xgb.train(
  params = best_params,
  data = dtrain_final,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain_final, eval = dvalid_final),
  early_stopping_rounds = 50,
  verbose = 1
)

cat("Final model best iteration:", final_model$best_iteration, "\n")

# -------------------- 12. SHAP analysis --------------------
cat("\n============================================================\n")
cat("8. SHAP ANALYSIS\n")
cat("============================================================\n")

shap_idx <- seq_len(nrow(scaled_final$train))
if (!is.null(N_SHAP_SAMPLES) && N_SHAP_SAMPLES < length(shap_idx)) {
  set.seed(SEED)
  shap_idx <- sample(shap_idx, N_SHAP_SAMPLES, replace = FALSE)
}

dshap <- xgb.DMatrix(as.matrix(scaled_final$train[shap_idx, FEATURE_COLS, drop = FALSE]))
shap_values <- predict(final_model, dshap, predcontrib = TRUE)
shap_values <- as.matrix(shap_values)

bias_col <- which(colnames(shap_values) == "Bias")
if (length(bias_col) > 0) {
  shap_values <- shap_values[, -bias_col, drop = FALSE]
}

common_cols <- intersect(colnames(shap_values), FEATURE_COLS)
shap_values <- shap_values[, common_cols, drop = FALSE]
X_scaled <- as.data.frame(as.matrix(scaled_final$train[shap_idx, common_cols, drop = FALSE]))
colnames(X_scaled) <- common_cols

X_original <- as.data.frame(lapply(common_cols, function(feat) {
  X_scaled[[feat]] * scaled_final$sd[feat] + scaled_final$mean[feat]
}))
colnames(X_original) <- common_cols

# SHAP long format (in-memory only, not exported)
shap_long <- build_shap_long(X_original, shap_values, common_cols)

# -------------------- 13. SHAP dependence plots (ONLY OUTPUT) --------------------
cat("\n============================================================\n")
cat("9. GENERATING SHAP DEPENDENCE PLOTS\n")
cat("============================================================\n")

for (feat in common_cols) {
  dep_df <- data.frame(
    feature = X_original[[feat]],
    shap = shap_values[, feat]
  )
  
  # Compute equal-interval 95% CI (in-memory, not exported)
  ci_df <- calc_equal_interval_ci(
    dep_df,
    x_col = "feature",
    shap_col = "shap",
    n_intervals = N_CI_INTERVALS
  )
  
  # Compute thresholds (in-memory, not exported)
  thresholds <- find_thresholds(
    dep_df$feature,
    dep_df$shap,
    span = LOESS_SPAN
  )
  
  # Plot
  p <- ggplot(dep_df, aes(x = feature, y = shap)) +
    geom_point(alpha = 0.3, size = 0.8, color = "gray40") +
    geom_smooth(method = "loess", span = LOESS_SPAN, se = TRUE, color = "blue") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(x = feat, y = "SHAP value") +
    theme_bw()
  
  if (length(thresholds) > 0) {
    for (tp in thresholds[seq_len(min(2, length(thresholds)))]) {
      p <- p + geom_vline(xintercept = tp, linetype = "dotdash", color = "red")
    }
  }
  
  ggsave(
    file.path(OUTPUT_DIR, paste0("SHAP_dependence_", feat, ".png")),
    p,
    width = 6,
    height = 4,
    dpi = 150
  )
  
  cat("  ✅ SHAP_dependence_", feat, ".png\n", sep = "")
}

cat("\n============================================================\n")
cat("Analysis completed successfully.\n")
cat("All SHAP dependence plots saved to:", normalizePath(OUTPUT_DIR, winslash = "/", mustWork = FALSE), "\n")
cat("============================================================\n")