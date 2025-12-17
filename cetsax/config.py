"""
Configuration and constants for CETSA EC50/KD modelling.
Dynamically loaded from config.yaml.
"""
# BSD 3-Clause License
#
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
# Email: mishraabhinav36@gmail.com
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Abhinav Mishra nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import yaml
from pathlib import Path

# 1. Locate config.yaml
# Assumes config.yaml is in the project root (parents[1] relative to cetsax/)
# You can adjust this logic if your project structure differs.
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_yaml_config(path: Path):
    """Safe load the yaml config."""
    if not path.exists():
        # Fallback: Try looking in current working directory
        path = Path("config.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find 'config.yaml' at {CONFIG_PATH} or current directory."
            )

    with open(path, "r") as f:
        return yaml.safe_load(f)


# 2. Load the Config
try:
    _cfg = load_yaml_config(CONFIG_PATH)
    _exp = _cfg.get("experiment", {})
except Exception as e:
    print(f"[WARNING] Failed to load config.yaml: {e}")
    print("Using hardcoded defaults for safety.")
    _exp = {}  # This will trigger the .get() defaults below

# 3. Map YAML values to Python Constants
# We use .get() with defaults to prevent crashes if yaml keys are missing

# Dose columns
DOSE_COLS = _exp.get("dose_columns", [
    "3.81e-06", "1.526e-05", "6.104e-05", "0.00024414",
    "0.00097656", "0.00390625", "0.015625", "0.0625", "0.25", "1"
])

# QC thresholds
_qc = _exp.get("qc_thresholds", {})
QC_MIN_UNIQUE_PEPTIDES = _qc.get("min_unique_peptides", 3)
QC_MIN_PSMS = _qc.get("min_psms", 15)
QC_MIN_COUNTNUM = _qc.get("min_countnum", 8)

# Column names
_cols = _exp.get("column_names", {})
ID_COL = _cols.get("id", "id")
COND_COL = _cols.get("condition", "condition")
SUM_UNIPEPS_COL = _cols.get("sum_unipeps", "sumUniPeps")
SUM_PSMS_COL = _cols.get("sum_psms", "sumPSMs")
COUNTNUM_COL = _cols.get("countnum", "countNum")

# Deep