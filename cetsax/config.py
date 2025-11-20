"""
Configuration and constants for CETSA EC50/KD modelling.
"""

# Dose columns in the CSV
DOSE_COLS = [
    "3.81e-06",
    "1.526e-05",
    "6.104e-05",
    "0.00024414",
    "0.00097656",
    "0.00390625",
    "0.015625",
    "0.0625",
    "0.25",
    "1",
]

# QC thresholds
QC_MIN_UNIQUE_PEPTIDES = 3
QC_MIN_PSMS = 15
QC_MIN_COUNTNUM = 8

# Column names
ID_COL = "id"
COND_COL = "condition"
SUM_UNIPEPS_COL = "sumUniPeps"
SUM_PSMS_COL = "sumPSMs"
COUNTNUM_COL = "countNum"
