# =====================================================================
# GR4H benchmark (airGR), calibrated on KGE: the control variant of
# manuscript Section 4.6 and Supplementary Information Section S6.
#
# Identical to scripts/gr4h/run_gr4h_airgr.R except for the calibration
# criterion (ErrorCrit_KGE instead of ErrorCrit_NSE) and the names of the
# output files.
#
# Input: data/gr4h/serie_horaria_gr4h.csv, the hourly catchment-mean
#   series built from the canonical dataset data/dataset_58585000_telem.h5
#   (rebuilt and checked by scripts/gr4h/make_gr4h_input.py):
#   P and E (mm/h) are area-weighted means over the 245 sub-catchments,
#   and Qmm (mm/h) is the observed discharge as a depth over the
#   3,117 km2 catchment area.
#   E is the hourly PET series that feeds the TOPMODEL layer (daily
#   Hargreaves-Samani PET divided by 24).
# Split: the bounds of the dataset splits used by the neural models.
#   Calibration 2021-01-01 00:00 to 2024-06-30 00:00 UTC, of which the
#   first 8,760 h (one year) are warm-up; test 2025-03-31 01:00 to
#   2025-12-31 00:00 UTC, warmed up over the 8,760 h before it.
# Output, in the folder given as the first argument (default:
#   outputs/retrain/gr4h under the repository root, not versioned):
#   gr4h_sim_teste_kge.csv (simulated and observed test discharge,
#   mm/h) and gr4h_parametros_kge.txt (X1 to X4 and the calibration
#   KGE, 4 decimals). The versioned results are in outputs/gr4h.
# The script only calibrates, simulates and exports. The test metrics
#   are computed by scripts/gr4h/audit_gr4h_common_mask.py with the
#   formulas and units of the neural evaluation (discharge in m3/s; FLV
#   in the modified log-tail form of Supplementary Section S11, not the
#   minimum-referenced form of Yilmaz et al., 2008), on the 6,323 scored
#   test hours of the neural models.
#
# Usage, from the repository root (the files in outputs/gr4h were
# produced with R 4.6.1 and airGR 1.7.8):
#   Rscript scripts/gr4h/run_gr4h_airgr_kge.R [output_folder]
# =====================================================================

suppressPackageStartupMessages({
  library(airGR)
})

## ---- 0. Paths -----------------------------------------------------------
# Repository root: two levels above this script when it runs through
# Rscript, otherwise the working directory.
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE),
                                        value = TRUE))
script_file <- gsub("~+~", " ", script_file, fixed = TRUE)
repo_root <- if (length(script_file) == 1L) {
  normalizePath(file.path(dirname(script_file), "..", ".."), winslash = "/")
} else {
  getwd()
}
input_csv <- file.path(repo_root, "data", "gr4h", "serie_horaria_gr4h.csv")
if (!file.exists(input_csv)) {
  stop("input not found: ", input_csv, " (run from the repository root)")
}
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[1] else file.path(repo_root, "outputs", "retrain", "gr4h")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## ---- 1. Data ------------------------------------------------------------
df <- read.csv(input_csv)
df$DatesR <- as.POSIXct(df$Dates, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
stopifnot(!any(is.na(df$DatesR)), !any(is.na(df$P)), !any(is.na(df$E)))
cat(sprintf("series: %s to %s | %d h | missing Q: %d\n",
            min(df$DatesR), max(df$DatesR), nrow(df), sum(is.na(df$Qmm))))

## ---- 2. Split of the neural models ---------------------------------------
date_train_ini <- as.POSIXct("2021-01-01 00:00", tz = "UTC")
date_train_fim <- as.POSIXct("2024-06-30 00:00", tz = "UTC")
date_test_ini  <- as.POSIXct("2025-03-31 01:00", tz = "UTC")
date_test_fim  <- as.POSIXct("2025-12-31 00:00", tz = "UTC")

idx_cal  <- which(df$DatesR >= date_train_ini & df$DatesR <= date_train_fim)
idx_test <- which(df$DatesR >= date_test_ini  & df$DatesR <= date_test_fim)
cat(sprintf("calibration: %d h | test: %d h\n", length(idx_cal), length(idx_test)))

## ---- 3. InputsModel (GR4H) ----------------------------------------------
InputsModel <- CreateInputsModel(
  FUN_MOD = RunModel_GR4H,
  DatesR  = df$DatesR,
  Precip  = df$P,
  PotEvap = df$E
)

## ---- 4. Warm-up and calibration on the training period -------------------
n_warmup <- min(length(idx_cal) - 1L, 365L * 24L)
idx_warm <- idx_cal[seq_len(n_warmup)]
idx_run  <- idx_cal[-seq_len(n_warmup)]

RunOptions_cal <- CreateRunOptions(
  FUN_MOD = RunModel_GR4H,
  InputsModel = InputsModel,
  IndPeriod_WarmUp = idx_warm,
  IndPeriod_Run    = idx_run
)

InputsCrit_cal <- CreateInputsCrit(
  FUN_CRIT = ErrorCrit_KGE,
  InputsModel = InputsModel,
  RunOptions  = RunOptions_cal,
  Obs = df$Qmm[idx_run]
)

CalibOptions <- CreateCalibOptions(
  FUN_MOD  = RunModel_GR4H,
  FUN_CALIB = Calibration_Michel
)

OutputsCalib <- Calibration_Michel(
  InputsModel  = InputsModel,
  RunOptions   = RunOptions_cal,
  InputsCrit   = InputsCrit_cal,
  CalibOptions = CalibOptions,
  FUN_MOD      = RunModel_GR4H
)
Param <- OutputsCalib$ParamFinalR
cat("Calibrated GR4H parameters (X1 to X4):\n"); print(Param)
cat(sprintf("Calibration KGE, unrounded: %.12f\n", OutputsCalib$CritFinal))

## ---- 5. Simulation of the TEST period ------------------------------------
n_warm_test <- min(idx_test[1] - 1L, 365L * 24L)
idx_warm_test <- (idx_test[1] - n_warm_test):(idx_test[1] - 1L)

RunOptions_test <- CreateRunOptions(
  FUN_MOD = RunModel_GR4H,
  InputsModel = InputsModel,
  IndPeriod_WarmUp = idx_warm_test,
  IndPeriod_Run    = idx_test
)
OutputsTest <- RunModel_GR4H(
  InputsModel = InputsModel,
  RunOptions  = RunOptions_test,
  Param       = Param
)

## ---- 6. Export simulated and observed discharge (metrics in Python) -------
out <- data.frame(
  Dates = format(df$DatesR[idx_test], "%Y-%m-%d %H:%M:%S"),
  Qsim_mm = OutputsTest$Qsim,
  Qobs_mm = df$Qmm[idx_test]
)
sim_file <- file.path(out_dir, "gr4h_sim_teste_kge.csv")
par_file <- file.path(out_dir, "gr4h_parametros_kge.txt")
write.csv(out, sim_file, row.names = FALSE)
writeLines(c(sprintf("X1=%.6g", Param[1]), sprintf("X2=%.6g", Param[2]),
             sprintf("X3=%.6g", Param[3]), sprintf("X4=%.6g", Param[4]),
             sprintf("KGE_cal=%.4f", OutputsCalib$CritFinal)),
           par_file)
cat(sprintf("\nWritten: %s and %s\n", sim_file, par_file))
