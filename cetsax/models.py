"""
models.py
---------------
Mathematical models for ITDR CETSA curves.
Provides the 4-parameter logistic ITDR model function.

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

from __future__ import annotations

import numpy as np


def itdr_model(c: np.ndarray, E0: float, Emax: float, logEC50: float, h: float) -> np.ndarray:
    """
    4-parameter logistic ITDR model for CETSA:

        f(c) = E0 + (Emax - E0) / (1 + (EC50 / c)^h)

    where EC50 = 10 ** logEC50 (parameterized in log10 space for stability).

    Parameters
    ----------
    c : np.ndarray
        Concentration array.
    E0 : float
        Response at zero concentration.
    Emax : float
        Maximum response at infinite concentration.
    logEC50 : float
        Log10 of the concentration at half-maximum response.
    h : float
        Hill slope.
    Returns
    -------
    np.ndarray
        Response values at concentrations c.
    """
    EC50 = 10.0 ** logEC50
    c = np.asarray(c, dtype=float)
    c_safe = np.where(c <= 0, np.min(c[c > 0]) * 1e-3, c)
    return E0 + (Emax - E0) / (1.0 + (EC50 / c_safe) ** h)
