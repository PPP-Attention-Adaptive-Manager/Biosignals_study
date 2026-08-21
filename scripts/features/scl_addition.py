"""
scl_addition.py
================
SCL (Skin Conductance Level) = mean of the tonic EDA component.

This is NOT a new signal-processing pipeline. Your existing
biosignals_targets.py already isolates the tonic component to compute
eda_tonic_slope (its rate of change over the window). SCL is the LEVEL
of that exact same isolated signal — same input array, different
summary statistic.

INTEGRATION
-----------
Drop extract_scl() below into biosignals_targets.py. Call it right
after wherever you currently fit a slope to the tonic component,
reusing the SAME tonic_component array — do not re-isolate the tonic
component from raw EDA a second time, and do not apply a different
low-pass filter than the one already used for eda_tonic_slope. If SCL
and eda_tonic_slope come from differently-filtered signals, the two
become hard to compare and the "two views of the same isolated signal"
property breaks.

This makes SCL the 13th biosignal target available from Cog Lab
(alongside the existing 12), used here as an INPUT for Version 2,
not as a prediction target.
"""

import numpy as np


def extract_scl(tonic_component: np.ndarray) -> float:
    """
    tonic_component: the already-isolated tonic EDA array for this
                      window — the same array your existing
                      eda_tonic_slope computation already has in scope
                      before fitting a slope to it.

    Returns: SCL, the mean tonic level for this window.
    """
    return float(np.mean(tonic_component))
