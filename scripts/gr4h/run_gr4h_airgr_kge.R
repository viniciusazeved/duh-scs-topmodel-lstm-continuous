# =====================================================================
# F - Benchmark externo GR4H (airGR) - Paper 1 simulacao continua
# Versao preenchida (revisao2_paper1), baseada no template do pacote de
# revisao. TODOs resolvidos:
#   - dados: serie_horaria_gr4h.csv (gerada do h5 canonico
#     ablacao_skill/data/dataset_58585000_telem.h5; P/E lumped por media
#     ponderada de area das 245 ottobacias; Q em mm/h com area 3117 km2)
#   - split IDENTICO ao pipeline Python (limites lidos do proprio h5):
#     treino 2021-01-01 00:00 -> 2024-06-30 00:00 UTC
#     teste  2025-03-31 01:00 -> 2025-12-31 00:00 UTC
#   - PET horaria = a MESMA serie do TOPMODEL (C1: Hargreaves diaria/24)
# MUDANCA deliberada vs template: as metricas finais (NSE/KGE/FLV/FHV)
# sao computadas em Python com o codigo exato do paper (consolida_sim),
# porque a convencao de FLV do template (Yilmaz, rebaixada pelo minimo)
# difere da do paper (razao de somas de log na base 30%). Aqui so se
# calibra, simula e exporta sim/obs.
# =====================================================================

suppressPackageStartupMessages({
  library(airGR)
})

## ---- 1. Dados ---------------------------------------------------------
df <- read.csv("D:/TTD_SCS_LSTM/revisao2_paper1/serie_horaria_gr4h.csv")
df$DatesR <- as.POSIXct(df$Dates, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
stopifnot(!any(is.na(df$DatesR)), !any(is.na(df$P)), !any(is.na(df$E)))
cat(sprintf("serie: %s -> %s | %d h | NaN Q: %d\n",
            min(df$DatesR), max(df$DatesR), nrow(df), sum(is.na(df$Qmm))))

## ---- 2. Split identico ao do paper -------------------------------------
date_train_ini <- as.POSIXct("2021-01-01 00:00", tz = "UTC")
date_train_fim <- as.POSIXct("2024-06-30 00:00", tz = "UTC")
date_test_ini  <- as.POSIXct("2025-03-31 01:00", tz = "UTC")
date_test_fim  <- as.POSIXct("2025-12-31 00:00", tz = "UTC")

idx_cal  <- which(df$DatesR >= date_train_ini & df$DatesR <= date_train_fim)
idx_test <- which(df$DatesR >= date_test_ini  & df$DatesR <= date_test_fim)
cat(sprintf("calibracao: %d h | teste: %d h\n", length(idx_cal), length(idx_test)))

## ---- 3. InputsModel (GR4H) ---------------------------------------------
InputsModel <- CreateInputsModel(
  FUN_MOD = RunModel_GR4H,
  DatesR  = df$DatesR,
  Precip  = df$P,
  PotEvap = df$E
)

## ---- 4. Warm-up + calibracao no treino ----------------------------------
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
cat("Parametros GR4H calibrados (X1..X4):\n"); print(Param)

## ---- 5. Simulacao no periodo de TESTE -----------------------------------
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

## ---- 6. Exporta sim/obs (metricas na convencao do paper, em Python) -----
out <- data.frame(
  Dates = format(df$DatesR[idx_test], "%Y-%m-%d %H:%M:%S"),
  Qsim_mm = OutputsTest$Qsim,
  Qobs_mm = df$Qmm[idx_test]
)
write.csv(out, "D:/TTD_SCS_LSTM/revisao2_paper1/gr4h_sim_teste_kge.csv", row.names = FALSE)
writeLines(c(sprintf("X1=%.6g", Param[1]), sprintf("X2=%.6g", Param[2]),
             sprintf("X3=%.6g", Param[3]), sprintf("X4=%.6g", Param[4]),
             sprintf("KGE_cal=%.4f", OutputsCalib$CritFinal)),
           "D:/TTD_SCS_LSTM/revisao2_paper1/gr4h_parametros_kge.txt")
cat("\nOK: gr4h_sim_teste_kge.csv + gr4h_parametros_kge.txt exportados.\n")
