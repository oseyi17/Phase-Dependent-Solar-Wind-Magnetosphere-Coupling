"""
NOVEL ADDITIONS (retained):
  1. Transfer Entropy (kNN)         — Kraskov-Frenzel-Pompe estimator; nonlinear,
                                       directional information flow with surrogate
                                       null-distribution significance testing.
  2. Wavelet Coherence              — time-frequency resolved coupling (non-stationary)
  3. Granger Causality              — F-test for linear predictive coupling
  4. Partial Cross-Correlation      — Bz / Ey contributions conditioned on each other
  5. Bootstrap Confidence Bands     — block-bootstrap CIs for all lag estimates

Author  : S.E. OLUKANNI
Period  : 2014–2018  (cadence set via run_pipeline(target_freq=...): "5min" or "1min")
Sources : OMNIWeb (NASA), Kyoto WDC
Python  : 3.10+
Deps    : numpy, pandas, scipy>=1.7, matplotlib, statsmodels, tqdm
"""

from __future__ import annotations

import gc
import logging
import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.spatial import KDTree
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("storm_coupling")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 350,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    }
)

PHASE_COLORS: dict[str, str] = {
    "initial": "#2196F3",
    "main": "#F44336",
    "recovery": "#4CAF50",
}

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUTION-AGNOSTIC TIME CONSTANTS
# ------------------------------------------------------------------------------
# Every window in the published Methodology is defined in *physical minutes*,
# not in samples. The constants below encode those physical windows once; every
# lag/limit "step count" used downstream is derived from them by dividing by
# the actual cadence (dt_min) of the loaded data. This means the analysis can
# be re-run on 1-min OMNIWeb data (dt_min=1) instead of 5-min data (dt_min=5)
# without changing any methodological window — only the sample counts that
# represent it change automatically.
# ══════════════════════════════════════════════════════════════════════════════
GAP_FILL_MAX_MIN: float = 30.0        # Sec. 2.1: gaps <= 6 rows @5-min filled
KP_STEP_MIN: float = 180.0            # Sec. 2.1: Kp is 3-hourly, forward-filled
CCF_MAX_LAG_MIN: float = 360.0        # Sec. 2.3: CCF searched over 0-360 min
TE_MAX_LAG_MIN: float = 240.0         # Sec. 2.4: TE searched over tau=5-240 min
GC_MAX_LAG_MIN: float = 100.0         # Sec. 2.6: Granger lag orders 5-100 min
STORM_CCF_MAX_LAG_MIN: float = 180.0  # storm-by-storm CCF robustness scan range
PARTIAL_CCF_AR_MAX_MIN: float = 30.0  # Sec. 2.7: Bz regressed on Ey, p=6 lags @5-min
STORM_TE_MIN_MIN: float = 300.0       # min main-phase duration for storm-by-storm TE
                                       # (5 h, matches original min_n=60 @5-min cadence;
                                       # must stay > TE_MAX_LAG_MIN or te_lag_scan can't
                                       # build valid lagged embeddings for short storms)
STORM_CCF_MIN_MIN: float = 750.0      # min main-phase duration for storm-by-storm CCF
                                       # (12.5 h, matches original min_n=150 @5-min)


def _steps_from_minutes(minutes: float, dt_min: float) -> int:
    """Convert a physical-time window (minutes) to a sample-step count at the
    given cadence dt_min, rounding to the nearest integer step (minimum 1)."""
    return max(1, int(round(minutes / dt_min)))


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: p-value display helper
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_p(p: float, n_surrogates: int) -> str:
    """
    Format a p-value from surrogate testing.

    When p == 0.0 the true value is only known to be < 1/n_surrogates.
    Reporting '0.000' implies false precision; report '< 0.01' instead.
    """
    resolution = 1.0 / n_surrogates
    if p < resolution:
        return f"< {resolution:.3f}"
    return f"{p:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

_PHYS_MAX: dict[str, float] = {
    "Bz":   500.0,
    "Bx":   500.0,
    "By":   500.0,
    "Ey":   100.0,
    "Vsw": 2000.0,
    "SYMH": 9000.0,
    "AE":   9000.0,
    "Kp":   100.0,
    # BUGFIX: these four were missing entirely when the extended parameter
    # set (Np, Tp, Pdyn, Beta) was added, so OMNI's own fill/missing-data
    # codes for them (999.9, 9999999., 99.99, 999.99 respectively) passed
    # straight through as if they were real measurements. Thresholds below
    # are set comfortably BELOW each documented fill code but ABOVE any
    # plausible real extreme value, so genuine extreme storms are preserved
    # while the fill sentinel is reliably caught.
    "Np":   900.0,      # fill = 999.9;      real Np rarely exceeds ~150 /cc
    "Tp":   8.0e6,       # fill = 9999999.;   real Tp rarely exceeds ~5e6 K
    "Pdyn": 95.0,        # fill = 99.99;      real Pdyn rarely exceeds ~80 nPa
    "Beta": 900.0,       # fill = 999.99;     real beta rarely exceeds ~100
}


def _parse_omni_time(df: pd.DataFrame, has_minute: bool) -> pd.DatetimeIndex:
    base = pd.to_datetime(
        df["year"].astype(str) + "-" + df["doy"].astype(str).str.zfill(3),
        format="%Y-%j",
    )
    offset = (
        pd.to_timedelta(df["hour"] * 60 + df["minute"], unit="min")
        if has_minute
        else pd.to_timedelta(df["hour"], unit="h")
    )
    return base + offset


def _clean_fill_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, upper in _PHYS_MAX.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].abs() < upper, other=np.nan)

    if "Kp" in df.columns:
        kp_max = df["Kp"].max(skipna=True)
        if pd.notna(kp_max) and kp_max > 9.0:
            df["Kp"] /= 10.0
            log.info(
                "Kp ×10 format detected (raw max %.1f); divided by 10. New max: %.1f",
                kp_max, df["Kp"].max(),
            )
    return df


def _skip_header_lines(filepath: str) -> int:
    year_re = re.compile(r"^\s*20\d{2}(\s|$)")
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.strip() and year_re.match(line.strip()):
                return i
    return 0


def load_others_file(
    filepath: str,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load an OMNIWeb solar-wind/geomagnetic file (breg_others) at any cadence
    (e.g. 5-min or 1-min high-resolution OMNI2 format); cadence is handled
    downstream in merge_to_5min(target_freq=...), not here.

    `columns` gives the data-column names IN FILE ORDER, after the leading
    YYYY DOY HR MN columns (which are always assumed present). If omitted,
    defaults to EXTENDED_OMNI_COLUMNS below.

    IMPORTANT — column order depends on what you requested from OMNIWeb:
    OMNIWeb's high-resolution request form (https://omniweb.gsfc.nasa.gov/
    form/omni_min.html) outputs whichever parameters you tick in ITS OWN
    fixed master order, not the order you clicked them in. For the extended
    parameter set used in this study (Bx, By, Bz, flow speed, density,
    temperature, flow pressure, electric field, plasma beta, AE, SYM/H),
    OMNIWeb's master order is:

        YYYY DOY HR MN Bx By Bz Vsw Np Tp Pdyn Ey Beta AE SYMH

    which is exactly EXTENDED_OMNI_COLUMNS below. If your downloaded file
    does not match this — e.g. you omitted a field, or OMNIWeb's own order
    has changed — pass the correct `columns` list explicitly, or check the
    physical-range warnings this function prints after loading (a swapped
    Np/Tp/Beta column shows up immediately as an out-of-range warning).
    """
    columns = columns or EXTENDED_OMNI_COLUMNS
    skip = _skip_header_lines(filepath)

    # Auto-detect legacy (7 data columns, 2014-2016 submission) vs extended
    # (11 data columns, with Np/Tp/Pdyn/Beta) schema by counting whitespace-
    # separated fields on the first data row, so old files still load
    # correctly without the caller having to pass `columns` explicitly.
    if columns is EXTENDED_OMNI_COLUMNS:
        with open(filepath) as f:
            for _ in range(skip):
                next(f)
            first_data_line = f.readline()
        n_fields = len(first_data_line.split())
        n_data_cols = n_fields - 4  # minus YYYY DOY HR MN
        if n_data_cols == len(LEGACY_OMNI_COLUMNS):
            log.info(
                "load_others_file: detected %d data columns -> using legacy "
                "schema (%s). Pass columns=EXTENDED_OMNI_COLUMNS explicitly "
                "if this file actually has the extended parameters.",
                n_data_cols, LEGACY_OMNI_COLUMNS,
            )
            columns = LEGACY_OMNI_COLUMNS
        elif n_data_cols != len(EXTENDED_OMNI_COLUMNS):
            raise ValueError(
                f"load_others_file: file has {n_data_cols} data columns, "
                f"which matches neither LEGACY_OMNI_COLUMNS "
                f"({len(LEGACY_OMNI_COLUMNS)}) nor EXTENDED_OMNI_COLUMNS "
                f"({len(EXTENDED_OMNI_COLUMNS)}). Pass the correct `columns` "
                f"list explicitly. First data row: {first_data_line.strip()}"
            )

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=skip,
        header=None,
        names=["year", "doy", "hour", "minute"] + columns,
    )
    df.index = _parse_omni_time(df, has_minute=True)
    df.index.name = "time"
    df = _clean_fill_values(df)
    _validate_physical_ranges(df)
    keep = [c for c in
            ["Bz", "Vsw", "Np", "Tp", "Pdyn", "Ey", "Beta", "AE", "SYMH"]
            if c in df.columns]
    return df[keep]


# Default column schema for the extended (2014-2018) dataset. See the
# load_others_file docstring above for why this specific order is expected.
EXTENDED_OMNI_COLUMNS: list[str] = [
    "Bx", "By", "Bz", "Vsw", "Np", "Tp", "Pdyn", "Ey", "Beta", "AE", "SYMH",
]

# Legacy schema (2014-2016 submission, no plasma thermodynamic parameters).
LEGACY_OMNI_COLUMNS: list[str] = ["Bx", "By", "Bz", "Vsw", "Ey", "AE", "SYMH"]

# Physical range checks: (min, max, unit) used to sanity-check that columns
# haven't been mis-assigned (e.g. Np/Tp/Beta swapped) after loading.
_PHYSICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "Bz":   (-300.0, 300.0, "nT"),
    "Vsw":  (200.0, 3000.0, "km/s"),
    "Np":   (0.05, 200.0, "cm^-3"),
    "Tp":   (1.0e2, 6.0e6, "K"),
    "Pdyn": (0.01, 90.0, "nPa"),
    "Beta": (0.001, 100.0, "dimensionless"),
    "Ey":   (-100.0, 100.0, "mV/m"),
    "AE":   (0.0, 6000.0, "nT"),
    "SYMH": (-700.0, 200.0, "nT"),
}


def _validate_physical_ranges(df: pd.DataFrame) -> None:
    """
    Sanity-check loaded columns against known physical ranges for each
    variable. This is a safety net against a column-order mismatch (e.g. Np
    and Tp swapped) that would otherwise silently corrupt every downstream
    result — swapped columns typically show up as >50% of values falling
    outside the expected physical range, which this flags loudly.
    """
    for col, (lo, hi, unit) in _PHYSICAL_RANGES.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        frac_outside = float(((vals < lo) | (vals > hi)).mean())
        if frac_outside > 0.05:
            log.warning(
                "COLUMN VALIDATION WARNING: %s has %.1f%% of values outside "
                "the expected physical range [%.3g, %.3g] %s. This usually "
                "means the file's column order doesn't match what was "
                "assumed (check EXTENDED_OMNI_COLUMNS / the `columns` "
                "argument) — verify before trusting downstream results.",
                col, frac_outside * 100, lo, hi, unit,
            )


def load_kp_file(filepath: str) -> pd.DataFrame:
    """
    Load the 3-hourly Kp OMNIWeb file (breg_kp).

    Supported column layouts (auto-detected by column count):
        YYYY DOY HR Kp          (4 cols)
        YYYY DOY HR MN Kp       (5 cols)
    """
    skip = _skip_header_lines(filepath)
    raw = pd.read_csv(filepath, sep=r"\s+", skiprows=skip, header=None)
    n_cols = raw.shape[1]

    if n_cols == 4:
        raw.columns = ["year", "doy", "hour", "Kp"]
        raw.index = _parse_omni_time(raw, has_minute=False)
    elif n_cols == 5:
        raw.columns = ["year", "doy", "hour", "minute", "Kp"]
        raw.index = _parse_omni_time(raw, has_minute=True)
    else:
        raise ValueError(
            f"breg_kp: expected 4 or 5 columns, got {n_cols}. "
            f"First data row: {raw.iloc[0].tolist()}"
        )

    raw.index.name = "time"
    raw = _clean_fill_values(raw)
    return raw[["Kp"]]


def merge_to_5min(
    others: pd.DataFrame,
    kp: pd.DataFrame,
    target_freq: str = "5min",
) -> pd.DataFrame:
    """
    Align the 3-hourly Kp series with the solar-wind / geomagnetic DataFrame;
    return a single combined DataFrame at `target_freq` cadence.

    NOTE ON RESOLUTION: `target_freq` may be "5min" or "1min" (or any other
    OMNIWeb cadence). The Kp forward-fill limit (Sec. 2.1: Kp is 3-hourly,
    stepwise held) and the short-gap interpolation limit (Sec. 2.1: gaps of
    <= 6 rows @5-min, i.e. <= 30 min, are linearly interpolated) are both
    derived from the *physical* KP_STEP_MIN / GAP_FILL_MAX_MIN constants
    divided by the actual cadence, so they represent the same real-world
    time window regardless of target_freq.
    """
    dt_min   = int(round(pd.Timedelta(target_freq) / pd.Timedelta(minutes=1)))
    kp_limit = _steps_from_minutes(KP_STEP_MIN, dt_min)
    gap_limit = _steps_from_minutes(GAP_FILL_MAX_MIN, dt_min)

    others_5 = others.resample(target_freq).mean()

    t_start = min(others_5.index.min(), kp.index.min())
    t_end   = max(others_5.index.max(), kp.index.max())
    full_idx = pd.date_range(start=t_start, end=t_end, freq=target_freq)

    kp_5min = (
        kp.reindex(full_idx.union(kp.index))
          .sort_index()
          .ffill(limit=kp_limit)
          .reindex(full_idx)
    )

    combined = others_5.reindex(full_idx).join(kp_5min)
    key_cols = [c for c in ["Bz", "Ey", "Vsw", "Np", "Tp", "Pdyn", "Beta", "SYMH", "AE"] if c in others.columns]
    was_nan_before = combined[key_cols].isna()
    combined[key_cols] = combined[key_cols].interpolate(method="time", limit=gap_limit)
    # A row is "interpolated" if any key column was NaN before the fill and is
    # now filled (i.e. filled by interpolate, not still NaN because the gap
    # exceeded gap_limit). Used by interpolation_sensitivity_check() to answer
    # the referee's question of whether gap-filling changes the CCF/TE results.
    combined["was_interpolated"] = (
        was_nan_before.to_numpy() & combined[key_cols].notna().to_numpy()
    ).any(axis=1)

    n_total = len(combined)
    n_valid = combined.dropna().shape[0]
    pct = 100.0 * n_valid / n_total if n_total else 0.0
    log.info(
        "Merged shape: %s  (%.1f%% rows fully non-NaN)  Range: %s → %s",
        combined.shape, pct, combined.index[0], combined.index[-1],
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, resample_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Z-score the driver/response columns.

    `resample_freq` is optional and defaults to None: the DataFrame coming out
    of merge_to_5min() is already on a uniform grid at its native cadence, so
    re-resampling here is unnecessary and, if left hardcoded to "5min", would
    silently downsample 1-min data. Pass an explicit resample_freq only if you
    need to change cadence at this stage.
    """
    if resample_freq is not None:
        df = df.resample(resample_freq).mean()
    for col in ["Bz", "Ey", "SYMH", "AE"]:
        if col in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2b.  DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_data(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("DATA DIAGNOSTICS")
    log.info("=" * 60)
    log.info("  Rows       : %d", len(df))
    log.info("  Time range : %s  to  %s", df.index[0], df.index[-1])
    log.info("  Cadence    : %s", df.index.to_series().diff().mode()[0])

    expected = {
        "Bz":   (-50,   20,  "nT   — mostly −5 to +5, storms to −30"),
        "Ey":   (-10,   10,  "mV/m — storms up to ±5"),
        "Vsw":  (250,  900,  "km/s — typical 350–600"),
        "SYMH": (-300,  50,  "nT   — storms below −50"),
        "AE":   (0,   3000,  "nT   — quiet <200, storms >500"),
        "Kp":   (0,      9,  "index"),
    }

    log.info("  %-6s %9s %9s %9s %7s  Note", "Col", "Min", "Mean", "Max", "NaN%")
    log.info("  " + "-" * 65)
    for col, (lo, hi, note) in expected.items():
        if col not in df.columns:
            log.warning("  %-6s *** COLUMN MISSING ***", col)
            continue
        s = df[col].dropna()
        pct = 100.0 * df[col].isna().mean()
        warn = " <-- ALL NEAR ZERO — wrong column?" if len(s) and s.abs().max() < 0.01 else ""
        log.info(
            "  %-6s %9.2f %9.2f %9.2f %6.1f%%  %s%s",
            col, s.min(), s.mean(), s.max(), pct, note, warn,
        )

    log.info("\n  SYM-H threshold scan:")
    for thr in [-30, -50, -75, -100]:
        n = (df["SYMH"] < thr).sum()
        pct = 100.0 * n / len(df)
        log.info("    < %4d nT : %6d rows  (%.2f%%)", thr, n, pct)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STORM IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def identify_storms(
    df: pd.DataFrame,
    symh_thresh: float = -50.0,
    kp_thresh: float = 5.0,
    min_sep_hrs: int = 24,
    onset_search_hrs: int = 24,
) -> list[dict]:
    """
    Identify geomagnetic storms and define their three phases.

    INITIAL  — 6 h before the first SYM-H crossing of symh_thresh.
    MAIN     — threshold crossing → global SYM-H minimum.
    RECOVERY — SYM-H minimum → first return to −10 nT, capped at 72 h.
    """
    below = (df["SYMH"] <= symh_thresh) & (df["Kp"] >= kp_thresh)
    crossings = df.index[below.astype(int).diff().fillna(0) == 1].tolist()

    storms: list[dict] = []
    prev_onset: Optional[pd.Timestamp] = None

    for onset in crossings:
        if (
            prev_onset is not None
            and (onset - prev_onset).total_seconds() / 3600 < min_sep_hrs
        ):
            continue

        search_end  = onset + pd.Timedelta(hours=onset_search_hrs)
        main_window = df.loc[onset:search_end, "SYMH"].dropna()
        if main_window.empty:
            continue

        min_idx   = main_window.idxmin()
        min_value = float(main_window.min())
        pre_start = max(onset - pd.Timedelta(hours=6), df.index[0])

        rec_limit = min_idx + pd.Timedelta(hours=72)
        rec_candidates = df.index[
            (df.index > min_idx)
            & (df.index <= rec_limit)
            & (df["SYMH"] >= -10.0)
        ]
        rec_end = rec_candidates[0] if len(rec_candidates) else rec_limit

        storms.append(
            {
                "pre_start": pre_start,
                "onset":     onset,
                "min_symh":  min_idx,
                "rec_end":   rec_end,
                "min_value": min_value,
            }
        )
        prev_onset = onset

    log.info(
        "Storm ID: found %d storms (SYM-H ≤ %.0f nT, Kp ≥ %.1f).",
        len(storms), symh_thresh, kp_thresh,
    )
    return storms


def label_phases(df: pd.DataFrame, storms: list[dict]) -> pd.Series:
    phase = pd.Series("quiet", index=df.index, dtype=str)
    for s in storms:
        phase.loc[s["pre_start"]: s["onset"]]    = "initial"
        phase.loc[s["onset"]:     s["min_symh"]] = "main"
        phase.loc[s["min_symh"]:  s["rec_end"]]  = "recovery"
    return phase


def phase_sampling_report(phases: pd.Series, dt_min: int = 5) -> pd.DataFrame:
    active_phases = ["initial", "main", "recovery"]
    counts = phases.value_counts()
    rows = []
    for ph in active_phases + ["quiet"]:
        n = int(counts.get(ph, 0))
        rows.append({"phase": ph, "n_rows": n, "hours": round(n * dt_min / 60, 1)})
    df_report = pd.DataFrame(rows)
    active_total = df_report.loc[df_report["phase"].isin(active_phases), "n_rows"].sum()
    df_report["pct_of_active"] = np.where(
        df_report["phase"].isin(active_phases),
        (df_report["n_rows"] / max(active_total, 1) * 100).round(1),
        np.nan,
    )
    log.info("Phase sampling summary:")
    log.info("  %-12s %8s %10s %18s", "Phase", "N rows", "Hours", "% of active")
    for _, r in df_report.iterrows():
        pct_str = f"{r['pct_of_active']:.1f}%" if pd.notna(r["pct_of_active"]) else "—"
        log.info("  %-12s %8d %10.1f %18s", r["phase"], r["n_rows"], r["hours"], pct_str)
    for _, r in df_report[df_report["phase"].isin(active_phases)].iterrows():
        if pd.notna(r["pct_of_active"]) and r["pct_of_active"] > 70.0:
            log.warning(
                "PHASE IMBALANCE: '%s' accounts for %.1f%% of active-phase rows.  "
                "Consider phase-stratified analysis.",
                r["phase"], r["pct_of_active"],
            )
    return df_report


def balance_phases(
    df: pd.DataFrame,
    phases: pd.Series,
    target_phase: str = "main",
    reference_phases: tuple[str, ...] = ("initial", "recovery"),
    strategy: str = "downsample",
    rng: Optional[np.random.Generator] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if rng is None:
        rng = np.random.default_rng(42)

    target_n = int((phases == target_phase).sum())
    keep_idx = list(df.index[phases == target_phase])

    for ph in reference_phases:
        ph_idx = df.index[phases == ph].tolist()
        n_ph   = len(ph_idx)
        if n_ph <= target_n or strategy == "report_only":
            keep_idx.extend(ph_idx)
            continue
        block_size      = max(1, int(np.sqrt(n_ph)))
        n_blocks_needed = int(np.ceil(target_n / block_size))
        n_blocks_avail  = max(1, n_ph - block_size + 1)
        starts = sorted(rng.choice(n_blocks_avail, n_blocks_needed, replace=False).tolist())
        sampled = []
        for s in starts:
            sampled.extend(ph_idx[s: s + block_size])
        keep_idx.extend(sampled[:target_n])
        log.info(
            "balance_phases: downsampled '%s' from %d → %d rows.",
            ph, n_ph, min(len(sampled), target_n),
        )

    keep_sorted = sorted(set(keep_idx))
    return df.loc[keep_sorted], phases.loc[keep_sorted]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SUPERPOSED EPOCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def superposed_epoch_analysis(
    df: pd.DataFrame,
    storms: list[dict],
    variables: list[str],
    window_before_hrs: int = 12,
    window_after_hrs: int = 48,
    dt_min: int = 5,
) -> dict:
    dt_min = int(round(dt_min))  # guard: dt_min must be a whole-minute cadence for
                                 # the floor-division below to yield integer steps
    steps_before = window_before_hrs * 60 // dt_min
    steps_after  = window_after_hrs  * 60 // dt_min
    epoch_len    = steps_before + steps_after + 1
    epoch_time   = np.arange(-steps_before, steps_after + 1) * dt_min / 60.0

    arrays: dict[str, list[np.ndarray]] = {v: [] for v in variables}
    for s in storms:
        idx_c = df.index.get_indexer([s["min_symh"]], method="nearest")[0]
        i0, i1 = idx_c - steps_before, idx_c + steps_after + 1
        if i0 < 0 or i1 > len(df):
            continue
        for v in variables:
            seg = df[v].iloc[i0:i1].values
            if len(seg) == epoch_len:
                arrays[v].append(seg)

    result: dict = {"epoch_time_hr": epoch_time}
    for v in variables:
        mat = np.array(arrays[v], dtype=float)
        result[v] = {
            "mean":   np.nanmean(mat, axis=0),
            "median": np.nanmedian(mat, axis=0),
            "std":    np.nanstd(mat, axis=0),
            "n":      mat.shape[0],
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  STORM-MAGNITUDE CLASSIFICATION  (referee Major #3)
# ══════════════════════════════════════════════════════════════════════════════

def classify_storms_by_magnitude(
    storms: list[dict],
    sigma_mult: float = 1.0,
) -> tuple[list[dict], dict]:
    """
    Classify storms into high / medium / quiet magnitude categories from the
    distribution of SYM-H minima across all storms:
        high    : min_value <= mean - sigma_mult*std   (most intense/negative)
        quiet   : min_value >= mean + sigma_mult*std   (least intense)
        medium  : everything in between

    NOTE: the referee's report says "mean ± 2σ". With sigma_mult=2.0 and only
    ~66 storms, a roughly-normal SYM-H-minimum distribution puts ~95% of
    storms inside +/-2sigma, so the "high" and "quiet" tails can end up nearly
    empty — check the logged category counts. sigma_mult=1.0 (default here)
    gives a more usable three-way split; change to 2.0 to match the referee's
    literal wording if you want the split exactly as requested and are fine
    with very few (or zero) storms in the outer categories.
    """
    vals = np.array([s["min_value"] for s in storms], dtype=float)
    mu, sigma = float(vals.mean()), float(vals.std())
    hi_thresh = mu - sigma_mult * sigma   # more negative SYM-H => stronger storm
    lo_thresh = mu + sigma_mult * sigma

    categorized = []
    for s in storms:
        v = s["min_value"]
        if v <= hi_thresh:
            cat = "high"
        elif v >= lo_thresh:
            cat = "quiet"
        else:
            cat = "medium"
        s2 = dict(s)
        s2["category"] = cat
        categorized.append(s2)

    counts = {c: sum(1 for s in categorized if s["category"] == c)
              for c in ("high", "medium", "quiet")}
    log.info(
        "Storm magnitude classification (sigma_mult=%.1f): mean=%.1f nT, std=%.1f nT "
        "-> thresholds high<=%.1f, quiet>=%.1f nT.  Counts: %s",
        sigma_mult, mu, sigma, hi_thresh, lo_thresh, counts,
    )
    if counts["high"] == 0 or counts["quiet"] == 0:
        log.warning(
            "Storm magnitude classification: one or more categories is empty "
            "at sigma_mult=%.1f (counts=%s). Consider a smaller sigma_mult or "
            "a percentile-based split (e.g. terciles) instead.",
            sigma_mult, counts,
        )
    return categorized, {"mean": mu, "std": sigma,
                          "hi_thresh": hi_thresh, "lo_thresh": lo_thresh}


def storm_peak_drivers(df: pd.DataFrame, storms: list[dict]) -> pd.DataFrame:
    """
    For each storm, extract the peak (most negative) Bz and peak Vsw observed
    between pre_start and min_symh (the driving interval), alongside the
    storm's SYM-H minimum. Used to correlate storm magnitude with driver
    strength (referee Major #3: "storm magnitudes-Bz-Vsw correlation").
    """
    rows = []
    for sid, s in enumerate(storms):
        win = df.loc[s["pre_start"]: s["min_symh"]]
        if win.empty:
            continue
        rows.append({
            "storm_id":    sid,
            "category":    s.get("category"),
            "min_symh":    s["min_value"],
            "peak_bz":     float(win["Bz"].min())  if "Bz"  in win else np.nan,
            "peak_vsw":    float(win["Vsw"].max()) if "Vsw" in win else np.nan,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4c.  CIR / ICME STORM-DRIVER CLASSIFICATION  (Reviewer 2: "requires analyzing")
# ══════════════════════════════════════════════════════════════════════════════

def expected_proton_temperature(vsw_km_s: np.ndarray) -> np.ndarray:
    """
    Expected ("normal solar wind expansion") proton temperature Texp(Vsw),
    from the linear fit of Elliott et al. (2012, JGR, doi:10.1029/2011JA017125):

        Texp [K] = 486.5 * Vsw [km/s] - 1.2476e5

    used following Richardson & Cane (1995, 2010) to identify anomalously
    cool solar wind intervals (Tp/Texp < 0.5) associated with ICME ejecta.
    Clipped at a small positive floor since the linear fit is not meant to
    be extrapolated to very low speeds.
    """
    return np.maximum(486.5 * vsw_km_s - 1.2476e5, 1.0e3)


def classify_storm_driver_type(
    df: pd.DataFrame,
    storms: list[dict],
    tp_texp_threshold: float = 0.5,
    beta_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Classify each storm's likely solar-wind driver as ICME-like (ejecta) or
    CIR-like (corotating interaction region) using two established, directly
    computable discriminators averaged over each storm's driving window
    (pre_start to min_symh):

      - Tp/Texp: the ratio of observed proton temperature to the expected
        temperature for normal solar-wind expansion at the observed speed
        (Elliott et al., 2012). Anomalously low Tp/Texp (< 0.5) is a
        long-established ICME/ejecta signature (Richardson & Cane, 1995,
        2010): reconnection-heated ejecta expand adiabatically and arrive
        cooler than ambient wind at the same speed.
      - Plasma beta: ICME ejecta typically have low beta (strong, ordered
        internal field dominating thermal pressure); CIRs, being compressed
        ambient wind, typically show elevated and more variable beta.

    Classification rule (storm-averaged over the driving window):
        ICME-like : mean(Tp/Texp) < tp_texp_threshold  AND mean(Beta) < beta_threshold
        CIR-like  : mean(Tp/Texp) >= 1.0               AND mean(Beta) >= beta_threshold
        ambiguous : neither pattern is clearly satisfied

    IMPORTANT LIMITATION: this is a deliberately simplified, two-criterion
    proxy, not a full replication of a published multi-criterion ICME/CIR
    catalogue (which typically also uses magnetic-field rotation smoothness,
    bidirectional suprathermal electron strahl, composition anomalies, and
    manual visual verification — none of which are available from the
    OMNIWeb parameters used in this study). Storms it calls "ambiguous"
    should be read as "not classifiable from these two criteria alone",
    not as a third physical category.
    """
    rows = []
    for sid, s in enumerate(storms):
        win = df.loc[s["pre_start"]: s["min_symh"]]
        if win.empty or "Tp" not in win.columns or "Vsw" not in win.columns:
            continue
        texp = expected_proton_temperature(win["Vsw"].values)
        tp_texp = win["Tp"].values / texp
        mean_tp_texp = float(np.nanmean(tp_texp))
        mean_beta = float(np.nanmean(win["Beta"])) if "Beta" in win.columns else np.nan
        mean_pdyn = float(np.nanmean(win["Pdyn"])) if "Pdyn" in win.columns else np.nan
        mean_np   = float(np.nanmean(win["Np"]))   if "Np"   in win.columns else np.nan

        if np.isnan(mean_tp_texp) or np.isnan(mean_beta):
            driver_type = "insufficient_data"
        elif mean_tp_texp < tp_texp_threshold and mean_beta < beta_threshold:
            driver_type = "ICME-like"
        elif mean_tp_texp >= 1.0 and mean_beta >= beta_threshold:
            driver_type = "CIR-like"
        else:
            driver_type = "ambiguous"

        rows.append({
            "storm_id": sid,
            "min_symh": s["min_value"],
            "mean_tp_texp": mean_tp_texp,
            "mean_beta": mean_beta,
            "mean_pdyn": mean_pdyn,
            "mean_np": mean_np,
            "driver_type": driver_type,
        })
    result = pd.DataFrame(rows)
    if len(result):
        counts = result["driver_type"].value_counts().to_dict()
        log.info(
            "CIR/ICME storm-driver classification (Tp/Texp<%.1f & beta<%.1f => "
            "ICME-like; Tp/Texp>=1.0 & beta>=%.1f => CIR-like): %s",
            tp_texp_threshold, beta_threshold, beta_threshold, counts,
        )
    return result


def main_phase_ccf_by_driver_type(
    df: pd.DataFrame,
    storms: list[dict],
    driver_type_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    max_lag_steps: int,
    dt_min: int,
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    """
    Re-run the main-phase cross-correlation (Section 2.4/3.4 methodology)
    separately for ICME-like vs CIR-like storms, to test whether the
    driver-response coupling lag differs by solar-wind driver type — the
    substantive question behind Reviewer 2's repeated CIR/CME request,
    rather than just reporting how many storms fall in each category.
    """
    storms_by_id = {sid: driver_type_df.set_index("storm_id")["driver_type"].get(sid)
                     for sid in range(len(storms))}
    rows = []
    for driver_type in ("ICME-like", "CIR-like"):
        subset = [s for sid, s in enumerate(storms)
                  if storms_by_id.get(sid) == driver_type]
        if len(subset) < 3:
            log.warning(
                "main_phase_ccf_by_driver_type: only %d storms classified as "
                "%s -- skipping (need >= 3 for a meaningful CCF).",
                len(subset), driver_type,
            )
            continue
        mask = label_phases(df, subset) == "main"
        sub_df = df.loc[mask]
        for drv, rsp in pairs:
            if len(sub_df) < 100:
                continue
            res = cross_correlation_lag(
                sub_df[drv].values, sub_df[rsp].values,
                max_lag_steps=max_lag_steps, dt_min=dt_min, n_bootstrap=n_bootstrap,
            )
            rows.append({
                "driver_type": driver_type, "n_storms": len(subset),
                "driver": drv, "response": rsp,
                "peak_lag_min": res["peak_lag_min"], "peak_r": res["peak_r"],
                "ci_low": res["ci_95"][0], "ci_high": res["ci_95"][1],
                "n_pts": res["n_points"],
            })
    result2 = pd.DataFrame(rows)
    if len(result2):
        log.info("\n--- CCF BY STORM-DRIVER TYPE (main phase) ---")
        for _, r in result2.iterrows():
            log.info(
                "  %-10s (N=%2d) | %s→%-5s  lag=%4d min  r=%+.3f",
                r["driver_type"], r["n_storms"], r["driver"], r["response"],
                r["peak_lag_min"], r["peak_r"],
            )
    return result2


def superposed_epoch_by_category(
    df: pd.DataFrame,
    categorized_storms: list[dict],
    variables: list[str],
    dt_min: int = 5,
    window_before_hrs: int = 12,
    window_after_hrs: int = 48,
) -> dict[str, dict]:
    """Run superposed_epoch_analysis() separately for each magnitude category."""
    out: dict[str, dict] = {}
    for cat in ("high", "medium", "quiet"):
        cat_storms = [s for s in categorized_storms if s["category"] == cat]
        if not cat_storms:
            log.warning("SEA-by-category: no storms in category '%s', skipping.", cat)
            continue
        out[cat] = superposed_epoch_analysis(
            df, cat_storms, variables=variables, dt_min=dt_min,
            window_before_hrs=window_before_hrs, window_after_hrs=window_after_hrs,
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CROSS-CORRELATION TIME-LAG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def cross_correlation_lag(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_steps: int = 72,
    dt_min: int = 5,
    n_bootstrap: int = 200,
    positive_lags_only: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)

    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    if np.nanstd(x) < 1e-6 or np.nanstd(y) < 1e-6:
        lags_tmp = (
            np.arange(0, max_lag_steps + 1)
            if positive_lags_only
            else np.arange(-max_lag_steps, max_lag_steps + 1)
        )
        return {
            "lags_min":     lags_tmp * dt_min,
            "ccf":          np.full(len(lags_tmp), np.nan),
            "peak_lag_min": 0,
            "peak_r":       0.0,
            "ci_95":        (0.0, 0.0),
            "n_points":     n,
        }

    
    lags = (
        np.arange(0, max_lag_steps + 1)
        if positive_lags_only
        else np.arange(-max_lag_steps, max_lag_steps + 1)
    )

    def _ccf(a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
        out = []
        for k in lags:
            try:
                if k == 0:
                    r = float(np.corrcoef(a, b)[0, 1])
                else:
                    r = float(np.corrcoef(a[:length - k], b[k:])[0, 1]) if length - k >= 2 else np.nan
            except Exception:
                r = np.nan
            out.append(r if np.isfinite(r) else np.nan)
        return np.array(out)

    ccf_vals  = _ccf(x, y, n)
    peak_idx  = int(np.nanargmax(np.abs(ccf_vals)))
    peak_lag_min = int(lags[peak_idx]) * dt_min
    peak_r    = float(ccf_vals[peak_idx])

    block_len = max(1, int(np.sqrt(n)))
    block_len = min(block_len, n // 2) if n >= 4 else 1
    n_blocks  = int(np.ceil(n / block_len))
    boot_peaks: list[int] = []
    _boot_hi = max(1, n - block_len + 1)
    for _ in range(n_bootstrap):
        starts = rng.integers(0, _boot_hi, n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        if len(idx) < 2:
            continue
        boot_r = _ccf(x[idx], y[idx], len(idx))
        boot_peaks.append(int(lags[np.nanargmax(np.abs(boot_r))]) * dt_min)

    if not boot_peaks:
        boot_peaks = [peak_lag_min]

    # ── Confidence interval: basic (reflected) bootstrap, THEN forced to ──────
    #    contain the point estimate.
    # The naive percentile CI (2.5th/97.5th percentile of the bootstrap peak
    # locations directly) can produce an interval that excludes the point
    # estimate itself whenever r(tau) is broad/flat near its maximum. The
    # basic/reflected bootstrap (Efron & Tibshirani, 1993) corrects much of
    # this, but is NOT itself a guarantee: it reflects the bootstrap
    # percentiles about the point estimate, which contains the point estimate
    # if and only if the point estimate lies within the [2.5th, 97.5th]
    # percentile range of its OWN bootstrap distribution -- and when the
    # full-sample estimate lands on a sharp, narrow local optimum that most
    # resampled replicates miss (a real, informative sign of a flat or
    # multimodal r(tau) curve near its maximum), that condition can fail.
    # We therefore take the union of the reflected interval with the point
    # estimate itself, which by construction always contains it; `ci_forced`
    # records whether this union step actually had to extend the interval,
    # so the caller can flag/report cases where it occurred rather than
    # silently widening the interval.
    lo = 2 * peak_lag_min - float(np.percentile(boot_peaks, 97.5))
    hi = 2 * peak_lag_min - float(np.percentile(boot_peaks, 2.5))
    lo, hi = min(lo, hi), max(lo, hi)
    ci_forced = not (lo <= peak_lag_min <= hi)
    lo = min(lo, peak_lag_min)
    hi = max(hi, peak_lag_min)
    valid_lo, valid_hi = float(lags.min()) * dt_min, float(lags.max()) * dt_min
    lo = max(lo, valid_lo)
    hi = min(hi, valid_hi)

    return {
        "lags_min":     lags * dt_min,
        "ccf":          ccf_vals,
        "peak_lag_min": peak_lag_min,
        "peak_r":       peak_r,
        "ci_95":        (lo, hi),
        "ci_forced":    ci_forced,
        "n_points":     n,
    }


def phase_lag_table(
    df: pd.DataFrame,
    phase_labels: pd.Series,
    pairs: list[tuple[str, str]],
    phases: tuple[str, ...] = ("initial", "main", "recovery"),
    min_r_threshold: float = 0.10,
    min_segment_hrs: float = 10.0,
    dt_min: int = 5,
    **kwargs,
) -> pd.DataFrame:
    min_pts = int(min_segment_hrs * 60 / dt_min)
    rows: list[dict] = []

    for phase in phases:
        sub    = df.loc[phase_labels == phase]
        n_rows = len(sub)
        if n_rows < min_pts:
            log.warning(
                "CCF SKIP %-10s: %d rows (%.1f h) — need ≥ %.0f h",
                phase, n_rows, n_rows * dt_min / 60, min_segment_hrs,
            )
            continue
        log.info("CCF %-10s: %d rows (%.0f h)", phase, n_rows, n_rows * dt_min / 60)

        for drv, rsp in pairs:
            res = cross_correlation_lag(
                sub[drv].values, sub[rsp].values, dt_min=dt_min, **kwargs
            )
            rows.append(
                {
                    "phase":        phase,
                    "driver":       drv,
                    "response":     rsp,
                    "peak_lag_min": res["peak_lag_min"],
                    "peak_r":       round(res["peak_r"], 3),
                    "ci_low":       res["ci_95"][0],
                    "ci_high":      res["ci_95"][1],
                    "ci_forced":    res.get("ci_forced", False),
                    "n_pts":        res["n_points"],
                    "reliable":     abs(res["peak_r"]) >= min_r_threshold,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["phase", "driver", "response", "peak_lag_min",
                     "peak_r", "ci_low", "ci_high", "ci_forced", "n_pts", "reliable"]
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRANSFER ENTROPY  (kNN estimator, surrogate significance)
# ══════════════════════════════════════════════════════════════════════════════
#
# SPEED ARCHITECTURE
# ------------------
# te_lag_scan() now subsamples to n_max (default 3 000) before the lag loop.
# This reduces the per-lag kNN cost from O(23 000) to O(3 000) — a ~50×
# reduction in tree-build and query time, bringing wall-clock time from ~6 h
# to ~10-15 min on 8 cores.
#
# The KFP estimator is asymptotically consistent; n = 3 000 gives TE values
# and surrogate p-values that are indistinguishable from n = 23 000 for the
# physical coupling strengths present in these data.
#
# PERFORMANCE: per-surrogate partial caching (tree_AC, tree_C static; only
# joint + tree_BC rebuilt per surrogate) is retained for further speedup.
# ══════════════════════════════════════════════════════════════════════════════

def interpolation_sensitivity_check(
    df: pd.DataFrame,
    phase_labels: pd.Series,
    pairs: list[tuple[str, str]],
    max_lag_steps: int,
    dt_min: int,
    min_r_threshold: float = 0.10,
    min_segment_hrs: float = 10.0,
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    """
    Referee comment (Major #1): does linear gap-filling change the reported
    CCF lags? Re-runs phase_lag_table() twice — once on the data as used in
    the main analysis (short gaps filled), once with every interpolated point
    reset to NaN (so cross_correlation_lag's own internal masking drops them,
    exactly as it already does for gaps that exceed the fill limit) — and
    reports the lag/r difference side by side for each phase x pair.

    Requires df to contain a "was_interpolated" column (added by merge_to_5min).
    """
    if "was_interpolated" not in df.columns:
        raise ValueError(
            "interpolation_sensitivity_check: df has no 'was_interpolated' "
            "column; re-run merge_to_5min() (this column is added there)."
        )

    key_cols = [c for c in ["Bz", "Ey", "Vsw", "SYMH", "AE"] if c in df.columns]
    df_raw = df.copy()
    df_raw.loc[df_raw["was_interpolated"], key_cols] = np.nan

    pct_interp = 100.0 * df["was_interpolated"].mean()
    log.info(
        "Interpolation sensitivity check: %.2f%% of rows were gap-filled overall.",
        pct_interp,
    )

    lag_interp = phase_lag_table(
        df, phase_labels, pairs, dt_min=dt_min, max_lag_steps=max_lag_steps,
        min_r_threshold=min_r_threshold, min_segment_hrs=min_segment_hrs,
        n_bootstrap=n_bootstrap,
    )
    lag_raw = phase_lag_table(
        df_raw, phase_labels, pairs, dt_min=dt_min, max_lag_steps=max_lag_steps,
        min_r_threshold=min_r_threshold, min_segment_hrs=min_segment_hrs,
        n_bootstrap=n_bootstrap,
    )

    merged = lag_interp.merge(
        lag_raw, on=["phase", "driver", "response"],
        suffixes=("_interpolated", "_raw_only"), how="outer",
    )
    merged["lag_diff_min"] = (
        merged["peak_lag_min_interpolated"] - merged["peak_lag_min_raw_only"]
    )
    merged["r_diff"] = merged["peak_r_interpolated"] - merged["peak_r_raw_only"]

    log.info("\n--- INTERPOLATION SENSITIVITY (interpolated vs raw-only) ---")
    for _, row in merged.iterrows():
        log.info(
            "  %-10s | %s→%-5s  lag: %4s → %4s min (Δ=%+.0f)   r: %+.3f → %+.3f (Δ=%+.3f)",
            row["phase"], row["driver"], row["response"],
            row["peak_lag_min_interpolated"], row["peak_lag_min_raw_only"],
            row["lag_diff_min"],
            row["peak_r_interpolated"], row["peak_r_raw_only"], row["r_diff"],
        )
    return merged


def _disp(col: str) -> str:
    """
    Map an internal dataframe column name to its consistent display form for
    figure titles/labels/captions. Internal code always indexes data by
    "SYMH" (a valid Python/pandas identifier), but every reader-facing
    string should read "SYM-H", never "SYMH" — this is the single place
    that mapping happens, so it can't drift out of sync between figures.
    """
    return {"SYMH": "SYM-H"}.get(col, col)


def _standardise_columns(arr: np.ndarray) -> np.ndarray:
    s = arr.std(axis=0)
    s[s < 1e-12] = 1.0
    return (arr - arr.mean(axis=0)) / s


    s = arr.std(axis=0)
    s[s < 1e-12] = 1.0
    return (arr - arr.mean(axis=0)) / s


def _knn_cmi_full(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    k: int = 5,
) -> tuple:
    """
    Kraskov-Frenzel-Pompe kNN CMI(A ; B | C) — full computation.
    Returns (cmi_bits, cmi_raw, eps, A_s, C_s, AC, tree_AC, tree_C).
    """
    if A.ndim == 1: A = A[:, np.newaxis]
    if B.ndim == 1: B = B[:, np.newaxis]
    if C.ndim == 1: C = C[:, np.newaxis]

    d_A, d_B = A.shape[1], B.shape[1]
    joint = _standardise_columns(np.hstack([A, B, C]))
    A_s   = joint[:, :d_A]
    B_s   = joint[:, d_A: d_A + d_B]
    C_s   = joint[:, d_A + d_B:]

    tree_joint = KDTree(joint)
    dists, _   = tree_joint.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, k] - 1e-10

    AC = np.hstack([A_s, C_s])
    BC = np.hstack([B_s, C_s])

    tree_AC = KDTree(AC)
    tree_BC = KDTree(BC)
    tree_C  = KDTree(C_s)

    n_AC = tree_AC.query_ball_point(AC, eps, p=np.inf, return_length=True) - 1
    n_BC = tree_BC.query_ball_point(BC, eps, p=np.inf, return_length=True) - 1
    n_C  = tree_C.query_ball_point(C_s, eps, p=np.inf, return_length=True) - 1

    mean_dg_AC = float(np.mean(digamma(np.maximum(n_AC, 0) + 1)))
    mean_dg_BC = float(np.mean(digamma(np.maximum(n_BC, 0) + 1)))
    mean_dg_C  = float(np.mean(digamma(np.maximum(n_C,  0) + 1)))

    cmi_nats = digamma(k) - mean_dg_AC - mean_dg_BC + mean_dg_C
    cmi_raw  = float(cmi_nats / np.log(2))
    cmi_bits = max(cmi_raw, 0.0)

    return cmi_bits, cmi_raw, eps, A_s, C_s, AC, tree_AC, tree_C


def _knn_cmi(A: np.ndarray, B: np.ndarray, C: np.ndarray, k: int = 5) -> float:
    cmi_bits, *_ = _knn_cmi_full(A, B, C, k=k)
    return cmi_bits


def _one_surrogate_correct(
    shift: int,
    full_source: np.ndarray,
    n_lag: int,
    A_s: np.ndarray,
    C_s: np.ndarray,
    AC: np.ndarray,
    tree_AC: KDTree,
    tree_C: KDTree,
    k: int,
) -> float:
    """
    Correct surrogate CMI for a single circular shift.
    Rebuilds joint + BC tree per surrogate (correct eps); AC and C trees cached.
    """
    B_raw = np.roll(full_source, shift)[:n_lag]
    b_std = B_raw.std()
    if b_std < 1e-12:
        b_std = 1.0
    B_s = ((B_raw - B_raw.mean()) / b_std)[:, np.newaxis]

    joint_surr = np.hstack([A_s, B_s, C_s])
    tree_joint_surr = KDTree(joint_surr)
    dists, _ = tree_joint_surr.query(joint_surr, k=k + 1, p=np.inf)
    eps_surr = dists[:, k] - 1e-10

    BC_surr = np.hstack([B_s, C_s])
    tree_BC_surr = KDTree(BC_surr)

    n_AC = tree_AC.query_ball_point(AC,      eps_surr, p=np.inf, return_length=True) - 1
    n_BC = tree_BC_surr.query_ball_point(BC_surr, eps_surr, p=np.inf, return_length=True) - 1
    n_C  = tree_C.query_ball_point(C_s,      eps_surr, p=np.inf, return_length=True) - 1

    cmi_nats = (
        digamma(k)
        - float(np.mean(digamma(np.maximum(n_AC, 0) + 1)))
        - float(np.mean(digamma(np.maximum(n_BC, 0) + 1)))
        + float(np.mean(digamma(np.maximum(n_C,  0) + 1)))
    )
    return cmi_nats / np.log(2)


def _te_surrogates_correct(
    full_source: np.ndarray,
    n_lag: int,
    A_s: np.ndarray,
    C_s: np.ndarray,
    AC: np.ndarray,
    tree_AC: KDTree,
    tree_C: KDTree,
    k: int,
    shifts: np.ndarray,
    n_jobs: int,
) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    import os as _os

    if n_jobs < 1:
        n_jobs = _os.cpu_count() or 1

    def _worker(shift: int) -> float:
        return _one_surrogate_correct(
            shift, full_source, n_lag, A_s, C_s, AC, tree_AC, tree_C, k,
        )

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(_worker, shifts.tolist()))

    return np.array(results, dtype=float)


def te_lag_scan(
    source: np.ndarray,
    target: np.ndarray,
    max_lag_steps: int = 48,
    k: int = 5,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    dt_min: int = 5,
    seed: int = 0,
    n_jobs: int = -1,
    show_progress: bool = True,
    n_min_reliable: int = 500,
    n_max: int = 3000,
) -> dict:
    """
    Scan TE(source → target) over lags 1…max_lag_steps.

    SPEED FIX
    ---------
    When len(source) > n_max, a random block-preserving subsample of n_max
    points is drawn before the lag loop.  This reduces the KDTree build cost
    by a factor of (n / n_max)^1.3 ≈ 50× for n = 23 000, n_max = 3 000.
    The KFP estimator is consistent: n = 3 000 gives reliable TE and p-values
    for the coupling strengths present in geomagnetic storm data.

    p-VALUE RESOLUTION
    ------------------
    The minimum non-zero p-value is 1/n_surrogates (e.g. 0.01 for 100
    surrogates).  p = 0.000 in the output means 0 surrogates exceeded the
    observed TE; the true bound is p < 1/n_surrogates.
    Use _fmt_p(p, n_surrogates) for display.

    Parameters
    ----------
    n_max : Maximum sample size fed to the kNN estimator.
            Default 3000.  Increase for final publication if runtime permits.
            Set to np.inf to disable subsampling.
    """
    rng = np.random.default_rng(seed)
    n   = len(source)

    # ── SPEED FIX: block-preserving RANDOM subsample ───────────────────────────
    # BUGFIX: the previous implementation computed a seeded random block-start
    # draw (`starts = rng.choice(...)`) but then never used it -- the actual
    # sample was a deterministic stride (`np.arange(0, n, stride)`), completely
    # independent of `seed`. This silently defeated any attempt to check
    # subsampling variability (e.g. a convergence sweep across seeds), and a
    # fixed stride can alias against periodic/quasi-periodic structure in the
    # data (storm spacing, diurnal cycles), which can distort the lag estimate
    # in ways unrelated to sample size. The corrected version below actually
    # uses the seeded rng to draw block starts, preserving local temporal
    # contiguity within each block (important for the lagged embedding) while
    # ensuring different seeds genuinely draw different subsamples.
    if n > n_max:
        block_len = max(1, int(np.sqrt(n_max)))
        n_blocks  = int(np.ceil(n_max / block_len))
        n_avail   = max(1, n - block_len + 1)
        starts    = rng.choice(n_avail, size=n_blocks,
                                replace=(n_blocks > n_avail))
        idx_sub   = np.unique(
            np.concatenate([np.arange(s, s + block_len) for s in starts])
        )
        idx_sub = idx_sub[idx_sub < n]
        if len(idx_sub) > n_max:
            keep    = rng.choice(len(idx_sub), size=n_max, replace=False)
            idx_sub = np.sort(idx_sub[keep])
        elif len(idx_sub) < n_max:
            remaining = np.setdiff1d(np.arange(n), idx_sub, assume_unique=True)
            extra_needed = n_max - len(idx_sub)
            if extra_needed > 0 and len(remaining) > 0:
                extra = rng.choice(
                    remaining, size=min(extra_needed, len(remaining)), replace=False,
                )
                idx_sub = np.sort(np.concatenate([idx_sub, extra]))
        source  = source[idx_sub]
        target  = target[idx_sub]
        n       = len(source)
        log.info(
            "te_lag_scan: subsampled to %d/%d points (n_max=%d, block_len=%d, "
            "seed=%d).", n, len(idx_sub), n_max, block_len, seed,
        )

    # ── Reliability check ──────────────────────────────────────────────────────
    reliable = n >= n_min_reliable
    if not reliable:
        log.warning(
            "TE reliability WARNING: n=%d < n_min_reliable=%d.  "
            "kNN estimates may be noisy.  Interpret p-values with caution.",
            n, n_min_reliable,
        )

    if max_lag_steps >= n:
        raise ValueError(
            f"te_lag_scan: max_lag_steps={max_lag_steps} >= n={n} samples "
            f"({max_lag_steps * dt_min:.0f} min lag window vs {n * dt_min:.0f} min "
            f"of data). Need n > max_lag_steps to build lagged embeddings at every "
            f"lag. Increase the caller's min_n filter (or shorten max_lag_steps for "
            f"this cadence) rather than passing a too-short segment here."
        )

    lags = np.arange(1, max_lag_steps + 1, dtype=int)

    te_vals    = np.empty(len(lags))
    surr_means = np.empty(len(lags))
    surr_stds  = np.empty(len(lags))
    surr_95    = np.empty(len(lags))
    p_vals     = np.empty(len(lags))

    # Pre-draw surrogate shifts
    _raw_min_shift    = max(max_lag_steps * 2, n // 10)
    min_shift_global  = min(_raw_min_shift, max(1, (n - 1) // 3))
    _shift_hi         = n - min_shift_global
    if _shift_hi <= min_shift_global:
        min_shift_global = 0
        _shift_hi = n
        log.warning(
            "te_lag_scan: n=%d very small; circular-shift guard relaxed.", n,
        )
    all_shifts = rng.integers(
        min_shift_global, _shift_hi,
        size=(len(lags), n_surrogates),
    ).astype(int)

    iterable = tqdm(
        enumerate(lags), total=len(lags),
        desc="TE lag scan", disable=not show_progress,
    )

    for idx, lag in iterable:
        n_lag = n - lag
        Y_t   = target[lag:][:n_lag]
        X_lag = source[:n_lag]
        C_arr = target[:n_lag]

        cmi_obs, cmi_raw, _eps_obs, A_s, C_s, AC, tree_AC, tree_C = _knn_cmi_full(
            Y_t, X_lag, C_arr, k=k
        )
        te_vals[idx] = cmi_obs

        surr = _te_surrogates_correct(
            full_source=source,
            n_lag=n_lag,
            A_s=A_s, C_s=C_s, AC=AC,
            tree_AC=tree_AC, tree_C=tree_C,
            k=k,
            shifts=all_shifts[idx],
            n_jobs=n_jobs,
        )
        surr_means[idx] = np.maximum(surr, 0).mean()
        surr_stds[idx]  = surr.std()
        surr_95[idx]    = max(np.percentile(surr, 95), 0.0)
        p_vals[idx]     = float(np.mean(surr >= cmi_raw))

    peak_idx = int(np.argmax(te_vals))
    return {
        "lags_steps":       lags,
        "lags_min":         lags * dt_min,
        "te":               te_vals,
        "te_surr_mean":     surr_means,
        "te_surr_std":      surr_stds,
        "te_surr_95":       surr_95,
        "p_values":         p_vals,
        "significant_mask": p_vals < alpha,
        "peak_lag_step":    int(lags[peak_idx]),
        "peak_lag_min":     int(lags[peak_idx]) * dt_min,
        "peak_te":          float(te_vals[peak_idx]),
        "peak_p":           float(p_vals[peak_idx]),
        "alpha":            alpha,
        "reliable":         reliable,
        "n_samples":        n,
        "n_surrogates":     n_surrogates,
    }


def transfer_entropy_knn(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
) -> float:
    """TE(source → target, τ=lag) via the KFP kNN estimator. Returns bits ≥ 0."""
    if len(source) < lag + 20 or len(target) < lag + 20:
        return 0.0
    n = len(source) - lag
    return _knn_cmi(
        A=target[lag: lag + n],
        B=source[:n],
        C=target[:n],
        k=k,
    )


def _align_pair(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    col_z: Optional[str] = None,
) -> tuple:
    """Drop NaN rows jointly; return aligned numpy arrays."""
    cols  = [col_x, col_y] + ([col_z] if col_z is not None else [])
    clean = df[cols].dropna()
    return tuple(clean[c].to_numpy(dtype=float) for c in cols)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  WAVELET COHERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _morlet_cwt(x: np.ndarray, scales: np.ndarray, w0: float = 6.0) -> np.ndarray:
    n = len(x)
    W = np.zeros((len(scales), n), dtype=complex)
    t = np.arange(n) - n // 2
    for i, s in enumerate(scales):
        psi  = (np.pi ** -0.25) * np.exp(1j * w0 * t / s) * np.exp(-(t ** 2) / (2 * s ** 2))
        norm = np.sqrt(2 * np.pi * s)
        W[i] = np.convolve(x, psi[::-1].conj(), mode="same") / norm
    return W


def _smooth_time_scale(W: np.ndarray, scales: np.ndarray, dj: float) -> np.ndarray:
    """
    Smooth a wavelet (cross-)spectrum in both time and scale, following
    Torrence & Compo (1998, Eq. 14) as implemented by Grinsted et al. (2004).
    This is REQUIRED before normalising to get coherence: without it, the
    normalised cross-spectrum is trivially ~1 at every point by construction
    (point-wise division of a quantity by itself), which is why the previous
    implementation (a single 5-sample boxcar in time only, no scale
    smoothing) produced a coherence map saturated at the top of the colour
    scale almost everywhere.

    Time smoothing: for the Morlet wavelet, convolve each scale's row with a
    Gaussian of width equal to that scale (in samples), consistent with the
    wavelet's own decorrelation length.
    Scale smoothing: convolve down each time-column with a boxcar of width
    0.6 (in units of log2-scale / dj), the standard Grinsted et al. choice.
    """
    n_scales, n_t = W.shape

    # --- time smoothing (scale-dependent Gaussian) ---
    time_smoothed = np.empty_like(W)
    for j, s in enumerate(scales):
        half = max(1, min(int(round(3 * s)), (n_t - 1) // 2))
        t = np.arange(-half, half + 1)
        gwin = np.exp(-0.5 * (t / s) ** 2)
        gwin /= gwin.sum()
        time_smoothed[j, :] = np.convolve(W[j, :], gwin, mode="same")

    # --- scale smoothing (boxcar of width 0.6 in log2-scale) ---
    win_len = max(1, int(round(0.6 / dj)))
    if win_len % 2 == 0:
        win_len += 1
    box = np.ones(win_len) / win_len
    scale_smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, box, mode="same"), 0, time_smoothed,
    )
    return scale_smoothed


def _ar1_coefficient(x: np.ndarray) -> float:
    """Lag-1 autocorrelation, used to build an AR(1) red-noise surrogate."""
    x = x - np.nanmean(x)
    denom = np.nansum(x[:-1] ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.nansum(x[:-1] * x[1:]) / denom, -0.99, 0.99))


def _ar1_surrogate(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Generate one AR(1) red-noise surrogate series of length n."""
    z = rng.standard_normal(n)
    s = np.empty(n)
    s[0] = z[0]
    for i in range(1, n):
        s[i] = alpha * s[i - 1] + np.sqrt(1 - alpha ** 2) * z[i]
    return s


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    dt: float = 5.0,
    min_period_min: float = 30.0,
    max_period_min: float = 360.0,
    n_scales: int = 40,
    n_surrogates: int = 100,
    seed: int = 0,
) -> dict:
    """
    Cross-wavelet transform and squared wavelet coherence |R(s,t)|^2 between
    x and y, following Torrence & Compo (1998) and Grinsted et al. (2004).

    Returns both the cross-wavelet power (XWT) and the coherence, each with
    an AR(1) Monte Carlo significance level, plus band-averaged coherence
    and mean phase for physical interpretation.
    """
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)

    dj = np.log2(max_period_min / min_period_min) / max(1, n_scales - 1)
    scales = np.logspace(
        np.log10(min_period_min / dt),
        np.log10(max_period_min / dt),
        n_scales,
    )
    Wx = _morlet_cwt(x, scales)
    Wy = _morlet_cwt(y, scales)

    Sxx = _smooth_time_scale(np.abs(Wx) ** 2, scales, dj)
    Syy = _smooth_time_scale(np.abs(Wy) ** 2, scales, dj)
    Sxy = _smooth_time_scale(Wx * np.conj(Wy), scales, dj)

    coherence = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-15)
    phase     = np.angle(Sxy)
    xwt_power = np.abs(Wx * np.conj(Wy))          # unsmoothed cross-wavelet power
    periods   = scales * dt

    # ── AR(1) Monte Carlo significance (Grinsted et al. 2004) ──────────────────
    rng = np.random.default_rng(seed)
    alpha_x, alpha_y = _ar1_coefficient(x), _ar1_coefficient(y)
    n = len(x)
    surr_xwt: list[np.ndarray] = []
    surr_coh: list[np.ndarray] = []
    for _ in range(n_surrogates):
        sx = _ar1_surrogate(n, alpha_x, rng)
        sy = _ar1_surrogate(n, alpha_y, rng)
        Wsx, Wsy = _morlet_cwt(sx, scales), _morlet_cwt(sy, scales)
        surr_xwt.append(np.abs(Wsx * np.conj(Wsy)))
        Ssxx = _smooth_time_scale(np.abs(Wsx) ** 2, scales, dj)
        Ssyy = _smooth_time_scale(np.abs(Wsy) ** 2, scales, dj)
        Ssxy = _smooth_time_scale(Wsx * np.conj(Wsy), scales, dj)
        surr_coh.append(np.abs(Ssxy) ** 2 / (Ssxx * Ssyy + 1e-15))
    xwt_sig95 = np.percentile(np.stack(surr_xwt), 95, axis=0)
    coh_sig95 = np.percentile(np.stack(surr_coh), 95, axis=0)

    bands = {"short": (30, 60), "medium": (60, 120), "long": (120, 360)}
    band_coherence: dict[str, np.ndarray] = {}
    for band_name, (p_lo, p_hi) in bands.items():
        mask = (periods >= p_lo) & (periods < p_hi)
        band_coherence[band_name] = (
            coherence[mask].mean(axis=0) if mask.any()
            else np.full(coherence.shape[1], np.nan)
        )

    mean_phase_deg = np.degrees(np.angle(np.exp(1j * phase).mean(axis=1)))

    # ── Cone of Influence (COI) ────────────────────────────────────────────────
    # Standard e-folding argument for the Morlet wavelet (Torrence & Compo 1998):
    # edge effects become significant beyond a time sqrt(2)*scale from either
    # boundary. Expressed as the *period* (minutes) above which each time step
    # falls inside the COI (i.e. is edge-affected and should be interpreted
    # cautiously / excluded).
    n_t = Wx.shape[1]
    t_idx = np.arange(n_t)
    edge_dist_steps = np.minimum(t_idx, n_t - 1 - t_idx).astype(float)
    coi_period_min = np.maximum(edge_dist_steps * dt / np.sqrt(2), 1e-6)

    return {
        "time_steps":     np.arange(len(x)),
        "period_min":     periods,
        "coherence":      coherence,
        "coherence_sig95": coh_sig95,
        "xwt_power":      xwt_power,
        "xwt_sig95":      xwt_sig95,
        "phase":          phase,
        "band_coherence": band_coherence,
        "mean_phase_deg": mean_phase_deg,
        "coi_period_min": coi_period_min,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7b.  STORM-BY-STORM VARIABILITY
# ══════════════════════════════════════════════════════════════════════════════

def storm_by_storm_te(
    df: pd.DataFrame,
    storms: list[dict],
    source_col: str,
    target_col: str,
    phase: str = "main",
    max_lag_steps: int = 48,
    k: int = 5,
    n_surrogates: int = 100,
    dt_min: int = 5,
    min_n: int = 150,
    n_max: int = 3000,
) -> pd.DataFrame:
    """
    Compute peak TE and peak lag independently for each storm event.
    n_max is passed through to te_lag_scan for consistent subsampling.
    """
    rows = []
    n_skipped = 0
    for sid, s in enumerate(storms):
        if phase == "main":
            t0, t1 = s["onset"], s["min_symh"]
        elif phase == "recovery":
            t0, t1 = s["min_symh"], s["rec_end"]
        else:
            t0, t1 = s["pre_start"], s["onset"]

        sub = df.loc[t0:t1, [source_col, target_col]].dropna()
        n   = len(sub)
        if n < min_n:
            n_skipped += 1
            continue

        src = sub[source_col].to_numpy(dtype=float)
        tgt = sub[target_col].to_numpy(dtype=float)
        try:
            res = te_lag_scan(
                src, tgt,
                max_lag_steps=max_lag_steps,
                k=k,
                n_surrogates=n_surrogates,
                dt_min=dt_min,
                show_progress=False,
                n_max=n_max,
            )
            rows.append({
                "storm_id":     sid,
                "onset":        s["onset"],
                "min_symh_val": s["min_value"],
                "n_pts":        n,
                "peak_te_bits": round(res["peak_te"], 5),
                "peak_lag_min": res["peak_lag_min"],
                "peak_p":       round(res["peak_p"], 4),
                "reliable":     res["reliable"],
            })
        except Exception as exc:
            log.warning("storm_by_storm_te: storm %d failed -- %s", sid, exc)
            n_skipped += 1

    result = pd.DataFrame(rows)
    if len(result):
        log.info(
            "storm_by_storm_te (%s→%s, %s): %d/%d storms  |  "
            "peak_te = %.4f ± %.4f bits  |  peak_lag = %.0f ± %.0f min",
            source_col, target_col, phase,
            len(result), len(storms),
            result["peak_te_bits"].mean(), result["peak_te_bits"].std(),
            result["peak_lag_min"].mean(), result["peak_lag_min"].std(),
        )
    else:
        log.warning(
            "storm_by_storm_te (%s→%s, %s): no storms qualified (min_n=%d; %d skipped).",
            source_col, target_col, phase, min_n, n_skipped,
        )
    return result


def storm_by_storm_ccf(
    df: pd.DataFrame,
    storms: list[dict],
    source_col: str,
    target_col: str,
    phase: str = "main",
    max_lag_steps: int = 72,
    dt_min: int = 5,
    min_n: int = 80,
) -> pd.DataFrame:
    rows = []
    n_skipped = 0
    for sid, s in enumerate(storms):
        if phase == "main":
            t0, t1 = s["onset"], s["min_symh"]
        elif phase == "recovery":
            t0, t1 = s["min_symh"], s["rec_end"]
        else:
            t0, t1 = s["pre_start"], s["onset"]

        sub = df.loc[t0:t1, [source_col, target_col]].dropna()
        n   = len(sub)
        if n < min_n:
            n_skipped += 1
            continue

        try:
            res = cross_correlation_lag(
                sub[source_col].to_numpy(dtype=float),
                sub[target_col].to_numpy(dtype=float),
                max_lag_steps=max_lag_steps,
                dt_min=dt_min,
                n_bootstrap=100,
            )
            rows.append({
                "storm_id":     sid,
                "onset":        s["onset"],
                "min_symh_val": s["min_value"],
                "n_pts":        n,
                "peak_lag_min": res["peak_lag_min"],
                "peak_r":       round(res["peak_r"], 4),
                "ci_low":       res["ci_95"][0],
                "ci_high":      res["ci_95"][1],
            })
        except Exception as exc:
            log.warning("storm_by_storm_ccf: storm %d failed -- %s", sid, exc)
            n_skipped += 1

    result = pd.DataFrame(rows)
    if len(result):
        log.info(
            "storm_by_storm_ccf (%s→%s, %s): %d/%d storms  |  "
            "lag = %.0f ± %.0f min  |  r = %.3f ± %.3f",
            source_col, target_col, phase,
            len(result), len(storms),
            result["peak_lag_min"].mean(), result["peak_lag_min"].std(),
            result["peak_r"].mean(), result["peak_r"].std(),
        )
    else:
        log.warning(
            "storm_by_storm_ccf (%s→%s, %s): no storms qualified (min_n=%d; %d skipped).",
            source_col, target_col, phase, min_n, n_skipped,
        )
    return result


def plot_storm_variability(
    te_var: pd.DataFrame,
    ccf_var: pd.DataFrame,
    source_col: str = "Bz",
    target_col: str = "SYMH",
) -> plt.Figure:
    from scipy.stats import pearsonr, spearmanr

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    pair_label = f"{_disp(source_col)} → {_disp(target_col)}"
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lbl in zip(axes.flat, panel_labels):
        ax.text(0.02, 0.96, lbl, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    ax = axes[0, 0]
    if not te_var.empty:
        ax.scatter(te_var["min_symh_val"], te_var["peak_lag_min"],
                   c="#7B1FA2", alpha=0.7, s=70, zorder=3, label="Per-storm TE lag")
        μ, σ = te_var["peak_lag_min"].mean(), te_var["peak_lag_min"].std()
        ax.axhline(μ, color="#7B1FA2", lw=1.8, ls="--", label=f"Mean = {μ:.0f} min")
        ax.axhspan(μ - σ, μ + σ, alpha=0.12, color="#7B1FA2", label=f"±1σ = {σ:.0f} min")
    ax.set_xlabel("SYM-H minimum (nT)", fontsize=12)
    ax.set_ylabel("Peak TE lag (min)", fontsize=12)
    ax.set_title(f"TE lag variability: {pair_label}", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    if not ccf_var.empty:
        ax.scatter(ccf_var["min_symh_val"], ccf_var["peak_lag_min"],
                   c="#0277BD", alpha=0.7, s=70, zorder=3, label="Per-storm CCF lag")
        μ, σ = ccf_var["peak_lag_min"].mean(), ccf_var["peak_lag_min"].std()
        ax.axhline(μ, color="#0277BD", lw=1.8, ls="--", label=f"Mean = {μ:.0f} min")
        ax.axhspan(μ - σ, μ + σ, alpha=0.12, color="#0277BD", label=f"±1σ = {σ:.0f} min")
    ax.set_xlabel("SYM-H minimum (nT)", fontsize=12)
    ax.set_ylabel("Peak CCF lag (min)", fontsize=12)
    ax.set_title(f"CCF lag variability: {pair_label}", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    if not te_var.empty and not ccf_var.empty:
        merged = te_var[["storm_id", "peak_lag_min"]].merge(
            ccf_var[["storm_id", "peak_lag_min"]], on="storm_id",
            suffixes=("_te", "_ccf"),
        )
        ax.scatter(merged["peak_lag_min_ccf"], merged["peak_lag_min_te"],
                   c="#424242", alpha=0.7, s=70, zorder=3)
        lim = [0, max(merged[["peak_lag_min_ccf", "peak_lag_min_te"]].max()) + 10]
        ax.plot(lim, lim, "k--", lw=1.0, label="TE = CCF lag")
        ax.axvline(merged["peak_lag_min_ccf"].mean(), color="#0277BD", lw=1.2, ls=":")
        ax.axhline(merged["peak_lag_min_te"].mean(),  color="#7B1FA2", lw=1.2, ls=":")
        # Referee comment #27c: quantify the TE-vs-CCF lag relationship.
        if len(merged) >= 3:
            r_p, p_p = pearsonr(merged["peak_lag_min_ccf"], merged["peak_lag_min_te"])
            r_s, p_s = spearmanr(merged["peak_lag_min_ccf"], merged["peak_lag_min_te"])
            ax.text(
                0.03, 0.97,
                f"Pearson r={r_p:+.2f} (p={p_p:.3f})\nSpearman ρ={r_s:+.2f} (p={p_s:.3f})",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="grey"),
            )
    ax.set_xlabel("Peak CCF lag (min)", fontsize=12)
    ax.set_ylabel("Peak TE lag (min)", fontsize=12)
    ax.set_title(f"TE vs CCF lag: {pair_label}", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    if not te_var.empty:
        ax.scatter(
            te_var["min_symh_val"], te_var["peak_te_bits"],
            c=["#E53935" if p < 0.05 else "#9E9E9E" for p in te_var["peak_p"]],
            alpha=0.8, s=70, zorder=3,
        )
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935",
                   markersize=9, label="p < 0.05"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
                   markersize=9, label="p ≥ 0.05"),
        ]
        ax.legend(handles=handles, fontsize=9)
        # Referee comment #27d: quantify TE amplitude vs storm intensity.
        if len(te_var) >= 3:
            r_p, p_p = pearsonr(te_var["min_symh_val"], te_var["peak_te_bits"])
            r_s, p_s = spearmanr(te_var["min_symh_val"], te_var["peak_te_bits"])
            ax.text(
                0.97, 0.97,
                f"Pearson r={r_p:+.2f} (p={p_p:.3f})\nSpearman ρ={r_s:+.2f} (p={p_s:.3f})",
                transform=ax.transAxes, fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="grey"),
            )
    ax.set_xlabel("SYM-H minimum (nT)", fontsize=12)
    ax.set_ylabel("Peak TE (bits)", fontsize=12)
    ax.set_title(f"TE amplitude vs storm intensity: {pair_label}", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(f"Storm-by-Storm Variability: {pair_label}", fontsize=15)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GRANGER CAUSALITY
# ══════════════════════════════════════════════════════════════════════════════

def granger_causality_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag_steps: int = 24,
    alpha: float = 0.05,
) -> dict:
    mask = np.isfinite(cause) & np.isfinite(effect)
    data = np.column_stack([effect[mask], cause[mask]])
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-9)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results_raw = grangercausalitytests(data, maxlag=max_lag_steps, verbose=False)
        except TypeError:
            # Newer statsmodels releases removed the `verbose` kwarg entirely.
            results_raw = grangercausalitytests(data, maxlag=max_lag_steps)

    lags, pvals, fstats = [], [], []
    for lag, res_dict in results_raw.items():
        lags.append(lag)
        pvals.append(res_dict[0]["ssr_ftest"][1])
        fstats.append(res_dict[0]["ssr_ftest"][0])

    lags   = np.array(lags)
    pvals  = np.array(pvals)
    fstats = np.array(fstats)

    return {
        "lags":        lags,
        "p_values":    pvals,
        "F_stats":     fstats,
        "significant": lags[pvals < alpha].tolist(),
        "min_p_lag":   int(lags[np.argmin(pvals)]),
        "min_p":       float(pvals.min()),
    }


def conditional_granger_causality_test(
    target: np.ndarray,
    conditioning_driver: np.ndarray,
    test_driver: np.ndarray,
    lag_order: int,
    alpha: float = 0.05,
) -> dict:
    """
    Conditional Granger causality: does `test_driver` carry predictive
    information about `target` beyond what `conditioning_driver` (and the
    target's own history) already provide?

    Implements exactly the nested-model F-test described in the manuscript
    Methods: a reduced VAR regressing target on its own lags and lags of the
    conditioning driver,

        target_t = sum_i a_i*target_(t-i) + sum_i b_i*cond_(t-i) + eps_t,

    against a full model additionally including lags of the test driver,

        target_t = sum_i a'_i*target_(t-i) + sum_i b'_i*cond_(t-i)
                   + sum_i c_i*test_(t-i) + eps'_t,

    both at the same lag order p = `lag_order`. Significance is assessed by
    an F-test on the reduction in residual sum of squares (RSS) from adding
    the test_driver lags, with (df1, df2) = (p, n - k_full) degrees of
    freedom, where k_full is the number of parameters in the full model
    (3p + 1, including the intercept).
    """
    from scipy.stats import f as f_dist

    mask = np.isfinite(target) & np.isfinite(conditioning_driver) & np.isfinite(test_driver)
    y, c, x = target[mask], conditioning_driver[mask], test_driver[mask]
    y = (y - y.mean()) / (y.std() + 1e-9)
    c = (c - c.mean()) / (c.std() + 1e-9)
    x = (x - x.mean()) / (x.std() + 1e-9)

    p = lag_order
    n = len(y) - p
    if n <= 3 * p + 5:
        raise ValueError(
            f"conditional_granger_causality_test: insufficient samples "
            f"(n={len(y)}) for lag_order={p}."
        )

    y_dep = y[p:]
    X_reduced_cols = [np.ones(n)]
    X_full_cols = [np.ones(n)]
    for i in range(1, p + 1):
        X_reduced_cols.append(y[p - i: len(y) - i])
        X_reduced_cols.append(c[p - i: len(c) - i])
    for col in X_reduced_cols:
        X_full_cols.append(col)
    for i in range(1, p + 1):
        X_full_cols.append(x[p - i: len(x) - i])

    X_reduced = np.column_stack(X_reduced_cols)
    X_full = np.column_stack(X_full_cols)

    beta_r, _, _, _ = np.linalg.lstsq(X_reduced, y_dep, rcond=None)
    beta_f, _, _, _ = np.linalg.lstsq(X_full, y_dep, rcond=None)
    rss_reduced = float(np.sum((y_dep - X_reduced @ beta_r) ** 2))
    rss_full = float(np.sum((y_dep - X_full @ beta_f) ** 2))

    k_full = X_full.shape[1]
    df1 = p
    df2 = n - k_full
    if df2 <= 0 or rss_full <= 0:
        raise ValueError("conditional_granger_causality_test: degenerate fit (check sample size/lag_order).")

    F = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    F = max(F, 0.0)
    p_value = float(f_dist.sf(F, df1, df2))

    return {
        "F_statistic": F, "df1": df1, "df2": df2, "p_value": p_value,
        "lag_order": p, "n_effective": n,
        "significant": p_value < alpha,
        "rss_reduced": rss_reduced, "rss_full": rss_full,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PARTIAL CROSS-CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def partial_cross_correlation(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    max_lag_steps: int = 72,
    dt_min: int = 5,
    ar_order: int = 6,
) -> dict:
    """
    Cross-correlation between x1 and y after partialling out x2 using a
    lag-distributed regression (avoids the zero-lag artefact of contemporaneous OLS).
    """
    from numpy.linalg import lstsq

    n = min(len(x1), len(x2), len(y))
    x1, x2, y = x1[:n], x2[:n], y[:n]
    p = min(ar_order, n // 10)

    if p < 1:
        log.warning(
            "partial_cross_correlation: n=%d too small; falling back to OLS.", n,
        )
        X = np.column_stack([x2, np.ones(n)])
        coef_x1, _, _, _ = lstsq(X, x1, rcond=None)
        coef_y,  _, _, _ = lstsq(X, y,  rcond=None)
        x1_partial = x1 - X @ coef_x1
        y_partial  = y  - X @ coef_y
    else:
        lag_mat  = np.column_stack([np.roll(x2, j) for j in range(1, p + 1)])[p:]
        x1_trim  = x1[p:]
        y_trim   = y[p:]
        X_lag    = np.column_stack([lag_mat, np.ones(len(x1_trim))])
        coef_x1, _, _, _ = lstsq(X_lag, x1_trim, rcond=None)
        coef_y,  _, _, _ = lstsq(X_lag, y_trim,  rcond=None)
        x1_partial = x1_trim - X_lag @ coef_x1
        y_partial  = y_trim  - X_lag @ coef_y

    return cross_correlation_lag(
        x1_partial, y_partial, max_lag_steps=max_lag_steps, dt_min=dt_min
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ASYMMETRY & DECOUPLING METRICS
# ══════════════════════════════════════════════════════════════════════════════

def decoupling_index(
    lag_AE_min: float,
    lag_SYMH_min: float,
    r_AE: float,
    r_SYMH: float,
) -> dict:
    delta_lag = abs(lag_AE_min - lag_SYMH_min)
    delta_r   = abs(abs(r_AE) - abs(r_SYMH))
    return {
        "delta_lag_min": delta_lag,
        "delta_r":       round(delta_r, 3),
        "D_index":       round(delta_lag * delta_r, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(
    df: pd.DataFrame,
    phase_labels: pd.Series,
    storms: list[dict],
    year: int = 2015,
) -> plt.Figure:
    sub = df[df.index.year == year]
    ph  = phase_labels[phase_labels.index.year == year]

    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.4), sharex=True)
    panel_meta = [
        ("Bz",   "IMF B$_z$ (nT)"),
        ("Ey",   "E$_y$ (mV m$^{-1}$)"),
        ("SYM-H", "SYM-H (nT)"),
        ("AE",   "AE (nT)"),
    ]
    for ax, (col, lab) in zip(axes, panel_meta):
        col_key = "SYMH" if col == "SYM-H" else col
        ax.plot(sub.index, sub[col_key], lw=0.5, color="0.3")
        for phase, color in PHASE_COLORS.items():
            ax.fill_between(
                sub.index, *ax.get_ylim(),
                where=ph.reindex(sub.index, fill_value="quiet") == phase,
                alpha=0.12, color=color,
                label=phase if col_key == "Bz" else "",
            )
        ax.set_ylabel(lab, fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.axhline(0, lw=0.5, color="k", ls="--")

    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    axes[2].axhline(-50, lw=1, color="red", ls=":", label="SYM-H = −50 nT")
    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle(f"Solar Wind and Geomagnetic Activity, {year} (1-minute cadence)",
                 fontsize=13)
    fig.tight_layout()
    return fig


def plot_sea(sea: dict, variables: list[str]) -> plt.Figure:
    t = sea["epoch_time_hr"]
    fig, axes = plt.subplots(len(variables), 1,
                             figsize=(7.4, 2.2 * len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]

    for ax, v in zip(axes, variables):
        m, s, n = sea[v]["mean"], sea[v]["std"], sea[v]["n"]
        ax.plot(t, m, lw=1.8, color="#1565C0", label=f"Mean (N = {n})")
        ax.fill_between(t, m - s, m + s, alpha=0.2, color="#1565C0", label="±1σ")
        ax.axvline(0, color="red", lw=1.2, ls="--", label="SYM-H min")
        ax.set_ylabel(v, fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Epoch time relative to SYM-H minimum (hours)", fontsize=11)
    fig.suptitle("Superposed Epoch Analysis", fontsize=13)
    fig.tight_layout()
    return fig


def plot_sea_by_category(
    sea_by_cat: dict[str, dict],
    variable: str,
    category_colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """
    Overlay superposed-epoch composites of `variable` for each storm-magnitude
    category (high/medium/quiet) on one axis, referee Major #3.
    """
    category_colors = category_colors or {
        "high": "#C62828", "medium": "#F9A825", "quiet": "#2E7D32",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat, sea in sea_by_cat.items():
        if variable not in sea:
            continue
        t = sea["epoch_time_hr"]
        m, s, n = sea[variable]["mean"], sea[variable]["std"], sea[variable]["n"]
        c = category_colors.get(cat, "grey")
        ax.plot(t, m, lw=2.2, color=c, label=f"{cat} (N={n})")
        ax.fill_between(t, m - s, m + s, alpha=0.15, color=c)
    ax.axvline(0, color="black", lw=1.2, ls="--", label="SYM-H min")
    units = {"Bz": "nT", "Vsw": "km s$^{-1}$", "Ey": "mV m$^{-1}$",
             "SYMH": "nT", "AE": "nT"}
    unit = units.get(variable, "")
    ylabel = f"{_disp(variable)} ({unit})" if unit else _disp(variable)
    ax.set_xlabel("Epoch time relative to SYM-H minimum (hours)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Superposed-Epoch Composite by Storm Magnitude Category: "
                 f"{_disp(variable)}", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_ccf_curve(ccf_result: dict, title: str = "") -> plt.Figure:
    """
    Plot a single representative CCF curve (r vs lag) with the peak marked
    and its bootstrap CI shaded, referee Major #4 ("include at least two
    representative cross-correlation plots to support the reported lag
    values").
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lags, ccf = ccf_result["lags_min"], ccf_result["ccf"]
    ax.plot(lags, ccf, lw=2.0, color="#1565C0")
    ax.axhline(0, color="grey", lw=0.8)
    peak_lag, peak_r = ccf_result["peak_lag_min"], ccf_result["peak_r"]
    ax.scatter([peak_lag], [peak_r], color="#C62828", zorder=5, s=90,
               label=f"Peak: lag={peak_lag} min, r={peak_r:+.3f}")
    lo, hi = ccf_result["ci_95"]
    ax.axvspan(lo, hi, color="#C62828", alpha=0.12, label=f"95% CI [{lo:.0f}, {hi:.0f}] min")
    ax.set_xlabel("Lag (min)", fontsize=12)
    ax.set_ylabel("Cross-correlation r", fontsize=12)
    ax.set_title(title or "Cross-Correlation vs Lag", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_lag_results(lag_df: pd.DataFrame) -> plt.Figure:
    pairs   = lag_df[["driver", "response"]].drop_duplicates()
    n_pairs = len(pairs)
    n_cols  = min(2, n_pairs)
    n_rows  = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows),
                              sharey=False, squeeze=False)
    axes_flat = axes.flat

    for ax, (_, row) in zip(axes_flat, pairs.iterrows()):
        sub    = lag_df[(lag_df["driver"] == row["driver"]) & (lag_df["response"] == row["response"])]
        colors = [PHASE_COLORS.get(p, "grey") for p in sub["phase"]]
        ax.barh(sub["phase"], sub["peak_lag_min"], color=colors, edgecolor="white")
        err_low  = np.clip(sub["peak_lag_min"] - sub["ci_low"],  0, None)
        err_high = np.clip(sub["ci_high"] - sub["peak_lag_min"], 0, None)
        ax.barh(sub["phase"], sub["peak_lag_min"],
                xerr=[err_low.values, err_high.values],
                color=colors, edgecolor="white", capsize=4, alpha=0.0)
        # Label placement: put the value PAST the end of the CI whisker (not
        # at the bar tip), with a white background box, so it never overlaps
        # the whisker line itself -- the overlap in the previous version is
        # what made the numbers look "clamped together with the line".
        x_max_needed = 0.0
        for _, r in sub.iterrows():
            label_x = max(r["peak_lag_min"], r["ci_high"]) + 2
            x_max_needed = max(x_max_needed, label_x)
            ax.text(label_x, sub["phase"].tolist().index(r["phase"]),
                    f'{r["peak_lag_min"]:.0f} min', va="center", fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
        ax.set_xlim(right=x_max_needed * 1.18 + 5)
        ax.set_title(f"{_disp(row['driver'])} → {_disp(row['response'])}", fontsize=12)
        ax.set_xlabel("Peak lag (minutes)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.axvline(0, color="k", lw=0.8, ls="--")

    for ax in list(axes_flat)[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle("Phase-Dependent Peak Lag Times (95% CI)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_te_scan(te_result: dict, title: str = "") -> plt.Figure:
    """
    Transfer entropy vs lag with surrogate null envelope.

    p-VALUE DISPLAY FIX: uses _fmt_p() so that p=0 is shown as
    '< 0.01' (for 100 surrogates) rather than the misleading '0.000'.
    """
    lags_min = te_result["lags_min"]
    te_obs   = te_result["te"]
    surr_95  = te_result["te_surr_95"]
    sig_mask = te_result["significant_mask"]
    n_surr   = te_result.get("n_surrogates", 100)
    peak_p   = te_result["peak_p"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.fill_between(lags_min, 0, surr_95,
                    alpha=0.25, color="grey", label="Surrogate 95th percentile")
    ax.plot(lags_min, te_obs, color="#7B1FA2", lw=1.8, label="Observed TE")
    ax.scatter(lags_min[sig_mask], te_obs[sig_mask],
               color="#E53935", zorder=5, s=40,
               label=f"Significant (p < {te_result.get('alpha', 0.05):.2f})")
    peak_min = te_result["peak_lag_min"]

    # p-value display fix
    p_display = _fmt_p(peak_p, n_surr)
    ax.axvline(
        peak_min, color="#E53935", lw=1.4, ls="--",
        label=f"Peak TE at τ = {peak_min} min  (p {p_display})",
    )
    ax.set_xlabel("Lag τ (minutes)", fontsize=11)
    ax.set_ylabel("Transfer Entropy (bits)", fontsize=11)
    ax.set_title(title or "Transfer Entropy", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)

    if not te_result.get("reliable", True):
        ax.text(
            0.5, 0.97,
            f"⚠  UNRELIABLE: n = {te_result.get('n_samples', '?')} < minimum.\n"
            "Significance results likely reflect estimation failure.",
            transform=ax.transAxes, va="top", ha="center", fontsize=8.5,
            color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                      edgecolor="#B71C1C", alpha=0.9),
        )

    fig.tight_layout()
    return fig


def plot_wavelet_band_coherence(
    wc: dict,
    time_index: pd.DatetimeIndex,
    title: str = "",
) -> plt.Figure:
    bands = wc.get("band_coherence", {})
    if not bands:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No band coherence data", ha="center", transform=ax.transAxes)
        return fig

    band_colors  = {"short": "#1565C0", "medium": "#E53935", "long": "#2E7D32"}
    band_labels  = {
        "short":  "30–60 min (immediate / DP2)",
        "medium": "60–120 min (ring current injection)",
        "long":   "120–360 min (recovery / long-term)",
    }
    n = min(min(len(v) for v in bands.values()), len(time_index))
    t = time_index[:n]

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.3), sharex=True)
    for ax, (band_name, color) in zip(axes, band_colors.items()):
        ax.plot(t, bands[band_name][:n], lw=1.4, color=color, label=band_labels[band_name])
        ax.axhline(0.5, color="k", lw=0.8, ls="--", label="Coherence = 0.5")
        ax.set_ylim(0, 1)
        ax.set_ylabel("|R²|", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle(title or "Band-Averaged Wavelet Coherence (Bz ↔ SYM-H)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_granger(gc_result: dict, title: str = "", dt_min: int = 5) -> plt.Figure:
    lags_min = gc_result["lags"] * dt_min
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    ax1.plot(lags_min, gc_result["F_stats"], color="#0277BD", lw=1.8)
    ax1.set_ylabel("F-statistic", fontsize=11)
    ax1.set_title(title or "Granger Causality", fontsize=13)
    ax1.tick_params(axis="both", labelsize=10)
    ax2.semilogy(lags_min, gc_result["p_values"], color="#2E7D32", lw=1.8)
    ax2.axhline(0.05, color="red", ls="--", lw=1, label="α = 0.05")
    ax2.set_ylabel("p-value (log scale)", fontsize=11)
    ax2.set_xlabel("Lag order (minutes)", fontsize=11)
    ax2.tick_params(axis="both", labelsize=10)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_wavelet_coherence(
    wc: dict,
    time_index: pd.DatetimeIndex,
    title: str = "",
) -> plt.Figure:
    """
    Two-panel figure following Grinsted et al. (2004): (a) cross-wavelet
    power (XWT), (b) squared wavelet coherence. Both panels show the AR(1)
    Monte Carlo 95% significance contour (thick black line), phase arrows
    (relative phase between the two series; pointing right = in phase,
    left = anti-phase, up/down = 90 degree lead/lag), and the cone of
    influence (hatched).

    PRINT SIZING: figsize is set to match a standard double-column journal
    figure width (~7.4 in = 188 mm) so that the fontsize values below (11 pt
    labels, 10 pt ticks) render at their stated point size when the figure
    is inserted at its native width, rather than being shrunk to near-
    illegible size by the typesetting process.
    """
    per = wc["period_min"] / 60.0
    n   = min(wc["coherence"].shape[1], len(time_index))
    coi_hr = wc.get("coi_period_min")
    coi_hr = coi_hr[:n] / 60.0 if coi_hr is not None else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.8), sharex=True)
    panel_specs = [
        (ax1, "(a)", wc["xwt_power"][:, :n], wc["xwt_sig95"][:, :n],
         "Cross-Wavelet Power", "plasma", None),
        (ax2, "(b)", wc["coherence"][:, :n], wc["coherence_sig95"][:, :n],
         "Squared Wavelet Coherence |R\u00b2|", "viridis", (0, 1)),
    ]

    # Sparse grid for phase arrows so the plot isn't cluttered.
    t_step = max(1, n // 30)
    p_step = max(1, len(per) // 12)

    for ax, label, field, sig95, subtitle, cmap, vlim in panel_specs:
        kwargs = dict(cmap=cmap, shading="auto")
        if vlim is not None:
            kwargs.update(vmin=vlim[0], vmax=vlim[1])
        im = ax.pcolormesh(time_index[:n], per, field, **kwargs)
        ax.set_yscale("log")
        ax.set_ylim(per.min(), per.max())
        ax.set_ylabel("Period (hours)", fontsize=11)
        ax.set_title(f"{label} {subtitle}", fontsize=12, loc="left")
        ax.tick_params(axis="both", labelsize=10)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=9)

        # AR(1) 95% significance contour.
        try:
            ax.contour(
                time_index[:n], per, field / (sig95 + 1e-15),
                levels=[1.0], colors="black", linewidths=1.2,
            )
        except Exception:
            pass

        # Phase arrows (Grinsted et al. convention): right = in phase,
        # left = anti-phase, down = X leads Y by 90 deg, up = Y leads X by
        # 90 deg. Drawn with a fixed length in point-space (not data-space)
        # so the arrow angle is not distorted by the log-scaled y-axis.
        phase = wc["phase"][:, :n]
        arrow_len_pts = 9
        for pi in range(0, len(per), p_step):
            for ti in range(0, n, t_step):
                ang = phase[pi, ti]
                dx, dy = arrow_len_pts * np.cos(ang), arrow_len_pts * np.sin(ang)
                ax.annotate(
                    "", xy=(time_index[ti], per[pi]), xycoords="data",
                    xytext=(-dx, -dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black",
                                     lw=0.7, alpha=0.6, shrinkA=0, shrinkB=0),
                )

        if coi_hr is not None:
            ax.plot(time_index[:n], coi_hr, color="white", lw=1.4, ls="-")
            ax.fill_between(
                time_index[:n], coi_hr, per.max(),
                where=coi_hr < per.max(),
                color="white", alpha=0.3, hatch="//", zorder=2,
            )

    ax2.set_xlabel("Date", fontsize=11)
    fig.suptitle(title or "Cross-Wavelet Transform and Coherence (Bz \u2194 SYM-H)",
                 fontsize=13)
    fig.tight_layout()
    return fig


def plot_decoupling_summary(lag_df: pd.DataFrame) -> dict[str, plt.Figure]:
    figs: dict[str, plt.Figure] = {}
    for drv in ["Bz", "Ey"]:
        sub  = lag_df[lag_df["driver"] == drv]
        x    = sub[sub["response"] == "SYMH"]["peak_lag_min"].values
        y    = sub[sub["response"] == "AE"]["peak_lag_min"].values
        ph   = sub[sub["response"] == "SYMH"]["phase"].values
        cols = [PHASE_COLORS.get(p, "grey") for p in ph]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(x, y, c=cols, s=120, zorder=3)
        for xi, yi, p in zip(x, y, ph):
            ax.annotate(p, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=10)
        if len(x) and len(y):
            lim = [min(x.min(), y.min()) - 10, max(x.max(), y.max()) + 10]
            ax.plot(lim, lim, "k--", lw=0.8, label="AE = SYM-H (no decoupling)")
        ax.set_xlabel("SYM-H lag (min)", fontsize=11)
        ax.set_ylabel("AE lag (min)", fontsize=11)
        ax.set_title(f"Lag Asymmetry: {_disp(drv)} → (SYM-H vs AE)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        figs[drv] = fig

    return figs


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    filepath_others: str,
    filepath_kp: str,
    output_dir: str = "results",
    target_freq: str = "5min",
    te_n_surrogates: int = 100,
    te_k: int = 5,
    te_max_lag_steps: Optional[int] = None,
    te_n_max: int = 3000,
) -> dict:
    """
    End-to-end analysis pipeline (real OMNIWeb CSV files only).

    Parameters
    ----------
    filepath_others :
        Path to the OMNIWeb file (Bz, Ey, Vsw, SYMH, AE) at `target_freq` cadence.
    filepath_kp :
        Path to the 3-hourly Kp OMNIWeb file.
    output_dir :
        Destination directory for figures and CSV summaries.
    target_freq :
        Cadence of the solar-wind/geomagnetic file, e.g. "5min" or "1min".
        Every downstream lag window (CCF, TE, Granger, partial-CCF, wavelet,
        gap-filling) is derived from this so the *physical* time windows in
        the Methodology are preserved regardless of cadence.
    te_n_surrogates :
        Surrogate count per lag.  Use 200–500 for publication; 100 for iteration.
    te_k :
        kNN neighbourhood size (default 5).
    te_max_lag_steps :
        Maximum TE lag, in samples at `target_freq` cadence. If None (default),
        derived from TE_MAX_LAG_MIN (240 min) so the physical search range
        matches the published methodology regardless of cadence.
    te_n_max :
        Maximum sample size for kNN estimation (default 3000).
        SPEED FIX: reduces TE runtime from ~6 h to ~10-15 min.
        Increase for final publication if compute permits.
    """
    dt_min = int(round(pd.Timedelta(target_freq) / pd.Timedelta(minutes=1)))

    # Derive every step-based window from the physical-time constants so the
    # methodology (in real minutes) is identical whether target_freq is
    # "5min" or "1min" — only the corresponding number of samples changes.
    ccf_max_lag_steps        = _steps_from_minutes(CCF_MAX_LAG_MIN, dt_min)
    te_max_lag_steps         = (
        _steps_from_minutes(TE_MAX_LAG_MIN, dt_min)
        if te_max_lag_steps is None else te_max_lag_steps
    )
    gc_max_lag_steps         = _steps_from_minutes(GC_MAX_LAG_MIN, dt_min)
    storm_ccf_max_lag_steps  = _steps_from_minutes(STORM_CCF_MAX_LAG_MIN, dt_min)
    partial_ccf_ar_order     = _steps_from_minutes(PARTIAL_CCF_AR_MAX_MIN, dt_min)
    storm_te_min_n           = _steps_from_minutes(STORM_TE_MIN_MIN, dt_min)
    # Guard: if te_max_lag_steps was overridden larger than the default
    # (e.g. widened past 240 min because a TE peak looked truncated at the
    # edge of the scan range), STORM_TE_MIN_MIN alone may no longer leave
    # enough margin above it — keep min_n at least max_lag + a 60-min margin.
    storm_te_min_n           = max(
        storm_te_min_n, te_max_lag_steps + _steps_from_minutes(60.0, dt_min),
    )
    storm_ccf_min_n          = _steps_from_minutes(STORM_CCF_MIN_MIN, dt_min)

    out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)
    log.info("Output directory: %s", out)
    log.info("=" * 70)
    log.info("Storm-Phase Time-Lag Analysis Pipeline")
    log.info("  Data cadence   = %s (dt_min=%.3g)", target_freq, dt_min)
    log.info("  CCF max lag    = %d steps (%.0f min)", ccf_max_lag_steps, ccf_max_lag_steps * dt_min)
    log.info("  TE max lag     = %d steps (%.0f min)", te_max_lag_steps, te_max_lag_steps * dt_min)
    log.info("  GC max lag     = %d steps (%.0f min)", gc_max_lag_steps, gc_max_lag_steps * dt_min)
    log.info("  TE n_max = %d  (speed fix; KFP consistent at this n)", te_n_max)
    log.info("  p-values displayed as '< 1/n_surr' when 0 surrogates exceed TE")
    log.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading solar-wind/geomagnetic file (%s cadence): %s", target_freq, filepath_others)
    others = load_others_file(filepath_others)
    log.info("Loading 3-h Kp file: %s", filepath_kp)
    kp = load_kp_file(filepath_kp)
    log.info("Merging to common %s grid …", target_freq)
    df_raw = merge_to_5min(others, kp, target_freq=target_freq)

    # ── Preprocess ────────────────────────────────────────────────────────────
    log.info("Preprocessing …")
    df = preprocess(df_raw)
    diagnose_data(df)

    # ── Storm identification ──────────────────────────────────────────────────
    log.info("Identifying storm events …")
    storms = identify_storms(df, symh_thresh=-50, kp_thresh=5)
    if not storms:
        symh_min = df["SYMH"].min()
        adaptive_thresh = round(symh_min * 0.6, 0)
        log.warning(
            "No storms at −50 nT. SYM-H min = %.1f nT.  "
            "Retrying with adaptive threshold = %.0f nT, Kp ≥ 3 …",
            symh_min, adaptive_thresh,
        )
        storms = identify_storms(df, symh_thresh=adaptive_thresh, kp_thresh=3)

    phases = label_phases(df, storms)
    for ph in ("initial", "main", "recovery"):
        n = (phases == ph).sum()
        log.info("  Phase [%-10s]: %6d rows  (%.0f h total)", ph, n, n * dt_min / 60)

    # ── Phase sampling diagnostics ─────────────────────────────────────────────
    phase_report = phase_sampling_report(phases, dt_min=dt_min)
    phase_report.to_csv(os.path.join(out, "phase_sampling_report.csv"), index=False)

    # ── Storm-magnitude classification (referee Major #3) ─────────────────────
    log.info("Classifying storms by magnitude (high/medium/quiet) …")
    storms_categorized, magnitude_stats = classify_storms_by_magnitude(
        storms, sigma_mult=1.0,
    )
    driver_df = storm_peak_drivers(df, storms_categorized)
    driver_df.to_csv(os.path.join(out, "storm_magnitude_bz_vsw.csv"), index=False)
    if len(driver_df) >= 3:
        from scipy.stats import pearsonr
        r_bz,  p_bz  = pearsonr(driver_df["min_symh"], driver_df["peak_bz"])
        r_vsw, p_vsw = pearsonr(driver_df["min_symh"], driver_df["peak_vsw"])
        log.info(
            "  Storm magnitude (SYM-H min) vs peak Bz : r=%+.3f (p=%.4f)", r_bz, p_bz,
        )
        log.info(
            "  Storm magnitude (SYM-H min) vs peak Vsw: r=%+.3f (p=%.4f)", r_vsw, p_vsw,
        )

    sea_by_cat = superposed_epoch_by_category(
        df, storms_categorized, variables=["Bz", "Vsw"], dt_min=dt_min,
    )
    for var in ("Bz", "Vsw"):
        fig_cat = plot_sea_by_category(sea_by_cat, var)
        fig_cat.savefig(os.path.join(out, f"sea_by_category_{var}.png"), dpi=300)
        plt.close(fig_cat)

    # ── CIR / ICME storm-driver classification (Reviewer 2) ────────────────────
    driver_type_df = pd.DataFrame()
    ccf_by_driver = pd.DataFrame()
    if {"Tp", "Beta"}.issubset(df.columns):
        log.info("Classifying storms by solar-wind driver type (ICME-like vs CIR-like) …")
        driver_type_df = classify_storm_driver_type(df, storms)
        driver_type_df.to_csv(os.path.join(out, "storm_driver_type.csv"), index=False)

        cross_tab = pd.DataFrame({
            sid: {"category": s.get("category"),
                  "driver_type": driver_type_df.set_index("storm_id")["driver_type"].get(sid, "n/a")}
            for sid, s in enumerate(storms_categorized)
        }).T
        cross_tab.to_csv(os.path.join(out, "storm_magnitude_vs_driver_type.csv"))
        log.info(
            "Magnitude category x driver type cross-tab:\n%s",
            pd.crosstab(cross_tab["category"], cross_tab["driver_type"]),
        )

        ccf_by_driver = main_phase_ccf_by_driver_type(
            df, storms, driver_type_df, pairs=[("Bz", "SYMH"), ("Ey", "AE")],
            max_lag_steps=ccf_max_lag_steps, dt_min=dt_min,
        )
        ccf_by_driver.to_csv(os.path.join(out, "ccf_by_driver_type.csv"), index=False)
    else:
        log.warning(
            "CIR/ICME classification skipped: Tp and/or Beta columns not present "
            "in the loaded data (only available with the extended OMNI parameter "
            "set — see load_others_file / EXTENDED_OMNI_COLUMNS)."
        )

    # ── SEA ───────────────────────────────────────────────────────────────────
    log.info("Superposed epoch analysis …")
    sea = superposed_epoch_analysis(df, storms, variables=["Bz", "Ey", "SYMH", "AE"],
                                     dt_min=dt_min)

    # ── Cross-correlation lags ─────────────────────────────────────────────────
    log.info("Phase-dependent lag analysis …")
    pairs  = [("Bz", "SYMH"), ("Ey", "SYMH"), ("Bz", "AE"), ("Ey", "AE")]
    lag_df = phase_lag_table(df, phases, pairs, n_bootstrap=100,
                              max_lag_steps=ccf_max_lag_steps, dt_min=dt_min,
                             min_r_threshold=0.10, min_segment_hrs=10.0)

    log.info("\n--- CCF SUMMARY (reliable rows only) ---")
    n_forced = 0
    for _, row in lag_df[lag_df["reliable"]].iterrows():
        forced_marker = " [CI extended to include point estimate]" if row.get("ci_forced") else ""
        if row.get("ci_forced"):
            n_forced += 1
        log.info(
            "  %-10s | %s→%-5s  lag=%4d min  r=%+.3f  CI=[%.0f, %.0f]%s",
            row["phase"], row["driver"], row["response"],
            row["peak_lag_min"], row["peak_r"], row["ci_low"], row["ci_high"],
            forced_marker,
        )
    if n_forced:
        log.info(
            "  NOTE: %d of %d rows required extending the reflected-bootstrap "
            "CI to include the point estimate -- this is a signal that r(tau) "
            "is flat or multimodal near its maximum for those pairs/phases, "
            "not a computation error. See phase_lag_table's 'ci_forced' column.",
            n_forced, lag_df["reliable"].sum(),
        )

    # ── Interpolation sensitivity check (referee Major #1) ─────────────────────
    log.info("\nInterpolation sensitivity check (interpolated vs raw-only) …")
    interp_sensitivity = interpolation_sensitivity_check(
        df, phases, pairs, max_lag_steps=ccf_max_lag_steps, dt_min=dt_min,
        min_r_threshold=0.10, min_segment_hrs=10.0, n_bootstrap=100,
    )
    interp_sensitivity.to_csv(
        os.path.join(out, "interpolation_sensitivity.csv"), index=False,
    )

    # ── Representative CCF plots (referee Major #4) ────────────────────────────
    log.info("Saving representative CCF curve plots …")
    for phase_name, drv, rsp in [("main", "Bz", "SYMH"), ("recovery", "Bz", "AE")]:
        sub = df.loc[phases == phase_name]
        if len(sub) < 2:
            continue
        ccf_curve = cross_correlation_lag(
            sub[drv].values, sub[rsp].values,
            max_lag_steps=ccf_max_lag_steps, dt_min=dt_min, n_bootstrap=100,
        )
        fig_ccf = plot_ccf_curve(
            ccf_curve, title=f"{phase_name.capitalize()} phase: {_disp(drv)} → {_disp(rsp)}",
        )
        fig_ccf.savefig(
            os.path.join(out, f"ccf_curve_{phase_name}_{drv}_{rsp}.png"), dpi=300,
        )
        plt.close(fig_ccf)

    # ── Transfer Entropy ───────────────────────────────────────────────────────
    log.info(
        "\nTransfer entropy scans (%d surrogates, n_max=%d) …",
        te_n_surrogates, te_n_max,
    )
    te_results: dict = {}
    for drv, rsp in [("Bz", "SYMH"), ("Ey", "AE")]:
        x, y = _align_pair(df[phases == "main"], drv, rsp)
        log.info("  %s→%s: n=%d main-phase samples (will subsample to ≤%d) …",
                 drv, rsp, len(x), te_n_max)
        te_res = te_lag_scan(
            x, y,
            max_lag_steps=te_max_lag_steps,
            k=te_k,
            n_surrogates=te_n_surrogates,
            dt_min=dt_min,
            show_progress=True,
            n_max=te_n_max,
        )
        te_results[(drv, rsp)] = te_res
        p_str = _fmt_p(te_res["peak_p"], te_n_surrogates)
        sig   = "*" if te_res["significant_mask"][
            te_res["lags_steps"] == te_res["peak_lag_step"]
        ].any() else ""
        log.info(
            "  %s→%s  peak TE = %.4f bits at τ = %d min  (p %s%s)",
            drv, rsp, te_res["peak_te"], te_res["peak_lag_min"], p_str, sig,
        )

    # ── Checkpoint: save TE results immediately ─────────────────────────────────
    # TE is by far the most expensive step (hours, at large n_max). Save it to
    # disk right away so a crash in any later, cheaper step (Granger, wavelet,
    # etc.) never loses this result again.
    te_summary_rows = []
    for (drv, rsp), te_res in te_results.items():
        pd.DataFrame({
            "lag_min": te_res["lags_min"], "te_bits": te_res["te"],
            "te_surr_95": te_res["te_surr_95"],
            "significant": te_res["significant_mask"],
        }).to_csv(os.path.join(out, f"te_curve_{drv}_{rsp}.csv"), index=False)
        te_summary_rows.append({
            "driver": drv, "response": rsp,
            "peak_lag_min": te_res["peak_lag_min"], "peak_te_bits": te_res["peak_te"],
            "peak_p": te_res["peak_p"], "n_max": te_n_max,
        })
    pd.DataFrame(te_summary_rows).to_csv(
        os.path.join(out, "te_summary.csv"), index=False,
    )
    log.info("Checkpoint: TE results saved to disk (te_curve_*.csv, te_summary.csv).")
    gc.collect()  # free the large kNN/surrogate working arrays before Granger

    # ── TE vs CCF interpretation ───────────────────────────────────────────────
    te_lag  = te_results[("Bz", "SYMH")]["peak_lag_min"]
    ccf_row = lag_df[
        (lag_df["driver"] == "Bz") & (lag_df["response"] == "SYMH") &
        (lag_df["phase"] == "main")
    ]
    if not ccf_row.empty:
        ccf_lag  = int(ccf_row["peak_lag_min"].values[0])
        lag_diff = te_lag - ccf_lag
        if lag_diff > 0:
            log.info(
                "\n  TE lag (%d min) > CCF lag (%d min) by %d min.\n"
                "  PHYSICAL INTERPRETATION: CCF captures the immediate linear "
                "response (magnetopause currents, DP2).  TE captures nonlinear "
                "energy accumulation (ring current injection).  Validate with "
                "medium-band (60–120 min) wavelet coherence.",
                te_lag, ccf_lag, lag_diff,
            )
        elif lag_diff < 0:
            log.info(
                "\n  TE lag (%d min) < CCF lag (%d min) — verify surrogate calibration.",
                te_lag, ccf_lag,
            )
        else:
            log.info(
                "\n  TE lag (%d min) == CCF lag (%d min) — consistent timescales.",
                te_lag, ccf_lag,
            )

    # ── Granger Causality ──────────────────────────────────────────────────────
    # Wrapped defensively: statsmodels' grangercausalitytests can raise
    # MemoryError on very large samples at high lag orders (observed on a
    # ~27,000-row main-phase series with max_lag_steps=100). A failure here
    # must NOT abort the whole pipeline and lose the TE/CCF work already
    # computed and checkpointed above -- it falls back to a capped random
    # subsample of the main-phase data, and if that still fails, logs the
    # failure and continues with gc_results left empty for that pair.
    log.info("\nGranger causality tests …")
    gc_results: dict = {}
    GRANGER_MAX_N = 15000  # Granger doesn't need the full main-phase pool for
                           # a stable F-test; this keeps the OLS design matrix
                           # a manageable size regardless of dataset length.
    for drv, rsp in [("Bz", "SYMH"), ("Ey", "AE")]:
        x, y = _align_pair(df[phases == "main"], drv, rsp)
        try:
            gc_res = granger_causality_test(x, y, max_lag_steps=gc_max_lag_steps)
        except MemoryError:
            log.warning(
                "Granger %s→%s: MemoryError at n=%d -- retrying on a random "
                "subsample of %d points.", drv, rsp, len(x), GRANGER_MAX_N,
            )
            gc.collect()
            try:
                rng = np.random.default_rng(0)
                if len(x) > GRANGER_MAX_N:
                    idx = np.sort(rng.choice(len(x), GRANGER_MAX_N, replace=False))
                    x, y = x[idx], y[idx]
                gc_res = granger_causality_test(x, y, max_lag_steps=gc_max_lag_steps)
            except Exception as e2:
                log.error(
                    "Granger %s→%s: failed even on subsample (%s). Skipping "
                    "this pair; all other results are unaffected.",
                    drv, rsp, e2,
                )
                continue
        except Exception as e:
            log.error(
                "Granger %s→%s: unexpected error (%s). Skipping this pair; "
                "all other results are unaffected.", drv, rsp, e,
            )
            continue
        gc_results[(drv, rsp)] = gc_res
        log.info(
            "  %s→%s  min-p lag = %d min   p = %.4e   sig lags: %s",
            drv, rsp,
            gc_res["min_p_lag"] * dt_min,
            gc_res["min_p"],
            gc_res["significant"],
        )
    gc.collect()

    # ── Conditional Granger Causality ───────────────────────────────────────────
    # Tests whether Bz carries predictive information about SYM-H beyond Ey
    # (and SYM-H's own history) -- this was described in the manuscript
    # Methods but had no corresponding implementation/call until now.
    cgc_res = None
    try:
        log.info("\nConditional Granger causality (Bz -> SYM-H | Ey) …")
        main_df_cgc = df[phases == "main"]
        cgc_lag_order = max(1, int(round(30.0 / dt_min)))  # 30-minute lag order
        cgc_res = conditional_granger_causality_test(
            main_df_cgc["SYMH"].values, main_df_cgc["Ey"].values,
            main_df_cgc["Bz"].values, lag_order=cgc_lag_order,
        )
        log.info(
            "  CGC: Bz -> SYM-H | Ey   F(%d, %d) = %.3f   p = %.3e   "
            "(lag order = %d steps = %d min, n_effective = %d)",
            cgc_res["df1"], cgc_res["df2"], cgc_res["F_statistic"],
            cgc_res["p_value"], cgc_res["lag_order"],
            cgc_res["lag_order"] * dt_min, cgc_res["n_effective"],
        )
    except Exception as e:
        log.error("Conditional Granger causality failed (%s); continuing without it.", e)
    if cgc_res is not None:
        pd.DataFrame([cgc_res]).to_csv(os.path.join(out, "cgc_result.csv"), index=False)

    # ── Partial Cross-Correlation ──────────────────────────────────────────────
    # Each remaining section is wrapped defensively: these are all much
    # cheaper than TE/Granger, but a failure in any one of them should still
    # not prevent the others (or the figures at the end) from being produced
    # and saved, given how much upstream work the pipeline represents.
    pcc_bz = pcc_ey = None
    try:
        log.info("\nPartial cross-correlations (main phase) …")
        main_clean = df[phases == "main"].dropna(subset=["Bz", "Ey", "SYMH", "AE"])
        pcc_bz = partial_cross_correlation(
            main_clean["Bz"].values, main_clean["Ey"].values, main_clean["SYMH"].values,
            max_lag_steps=ccf_max_lag_steps, dt_min=dt_min, ar_order=partial_ccf_ar_order,
        )
        pcc_ey = partial_cross_correlation(
            main_clean["Ey"].values, main_clean["Bz"].values, main_clean["SYMH"].values,
            max_lag_steps=ccf_max_lag_steps, dt_min=dt_min, ar_order=partial_ccf_ar_order,
        )
        log.info("  Bz→SYM-H (partial):  lag = %d min  r = %.3f",
                 pcc_bz["peak_lag_min"], pcc_bz["peak_r"])
        log.info("  Ey→SYM-H (partial):  lag = %d min  r = %.3f",
                 pcc_ey["peak_lag_min"], pcc_ey["peak_r"])
        if pcc_bz["peak_lag_min"] == 0:
            log.warning(
                "Partial CCF Bz→SYM-H: zero lag — possible residual multicollinearity. "
                "Interpret cautiously."
            )
    except Exception as e:
        log.error("Partial cross-correlation failed (%s); continuing without it.", e)

    # ── Storm-by-storm variability ─────────────────────────────────────────────
    te_var_bz = pd.DataFrame()
    ccf_var_bz = pd.DataFrame()
    try:
        log.info("\nStorm-by-storm TE and CCF variability …")
        te_var_bz = storm_by_storm_te(
            df, storms, source_col="Bz", target_col="SYMH", phase="main",
            max_lag_steps=te_max_lag_steps, k=te_k,
            n_surrogates=max(50, te_n_surrogates // 4),
            dt_min=dt_min, min_n=storm_te_min_n, n_max=te_n_max,
        )
        ccf_var_bz = storm_by_storm_ccf(
            df, storms, source_col="Bz", target_col="SYMH",
            phase="main", max_lag_steps=storm_ccf_max_lag_steps, dt_min=dt_min,
            min_n=storm_ccf_min_n,
        )
        te_var_bz.to_csv(os.path.join(out, "storm_by_storm_te_bz_symh.csv"), index=False)
        ccf_var_bz.to_csv(os.path.join(out, "storm_by_storm_ccf_bz_symh.csv"), index=False)
    except Exception as e:
        log.error("Storm-by-storm variability failed (%s); continuing without it.", e)

    # ── Wavelet Coherence ──────────────────────────────────────────────────────
    wc_res = None
    try:
        log.info("\nWavelet coherence (Bz–SYM-H, main phase) …")
        wc_sub = df[phases == "main"].dropna(subset=["Bz", "SYMH"])
        # NOTE: 2,000 is a fixed *sample count* per the Methodology (Sec. 2.8:
        # "first 2,000 contiguous main-phase data points"), not a time duration,
        # so it is intentionally left unscaled by cadence.
        n_wc   = min(2000, len(wc_sub))
        wc_res = wavelet_coherence(
            wc_sub["Bz"].values[:n_wc],
            wc_sub["SYMH"].values[:n_wc],
            dt=dt_min, min_period_min=30, max_period_min=360,
        )
    except Exception as e:
        log.error("Wavelet coherence failed (%s); continuing without it.", e)

    # ── Decoupling Index ───────────────────────────────────────────────────────
    log.info("\nDecoupling indices per phase …")
    for phase in ("initial", "main", "recovery"):
        sub_ph = lag_df[lag_df["phase"] == phase]
        for drv in ("Bz", "Ey"):
            r_ae   = sub_ph[(sub_ph["driver"] == drv) & (sub_ph["response"] == "AE")]
            r_symh = sub_ph[(sub_ph["driver"] == drv) & (sub_ph["response"] == "SYMH")]
            if r_ae.empty or r_symh.empty:
                continue
            di = decoupling_index(
                r_ae["peak_lag_min"].values[0],
                r_symh["peak_lag_min"].values[0],
                r_ae["peak_r"].values[0],
                r_symh["peak_r"].values[0],
            )
            log.info("  [%-10s] %s: Δlag=%d min  Δr=%.3f  D=%.2f",
                     phase, drv, di["delta_lag_min"], di["delta_r"], di["D_index"])

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Built incrementally with one try/except PER figure: a missing or failed
    # upstream result (e.g. Granger/wavelet failing for one pair) must only
    # skip that one figure, not abort every figure the pipeline would
    # otherwise still be able to produce and save.
    log.info("\nGenerating figures …")
    figs: dict[str, plt.Figure] = {}

    def _try_fig(name: str, builder):
        try:
            figs[name] = builder()
        except Exception as e:
            log.error("Figure '%s' failed (%s); skipping.", name, e)

    _try_fig("overview",   lambda: plot_overview(df, phases, storms))
    _try_fig("sea",        lambda: plot_sea(sea, ["Bz", "Ey", "SYMH", "AE"]))
    _try_fig("lag_chart",  lambda: plot_lag_results(lag_df))
    if ("Bz", "SYMH") in te_results:
        _try_fig("te_bz_symh", lambda: plot_te_scan(
            te_results[("Bz", "SYMH")], "Transfer Entropy: Bz → SYM-H (main phase)"))
    if ("Ey", "AE") in te_results:
        _try_fig("te_ey_ae", lambda: plot_te_scan(
            te_results[("Ey", "AE")], "Transfer Entropy: Ey → AE (main phase)"))
    if ("Bz", "SYMH") in gc_results:
        _try_fig("gc_bz_symh", lambda: plot_granger(
            gc_results[("Bz", "SYMH")], "Granger Causality: Bz → SYM-H (main phase)"))
    if wc_res is not None:
        _try_fig("wc_bz_symh", lambda: plot_wavelet_coherence(
            wc_res, wc_sub.index[:n_wc], "Wavelet Coherence: Bz ↔ SYM-H (main phase)"))
        _try_fig("wc_bands", lambda: plot_wavelet_band_coherence(
            wc_res, wc_sub.index[:n_wc], "Band-averaged Coherence: Bz ↔ SYM-H (main phase)"))
    if not te_var_bz.empty or not ccf_var_bz.empty:
        _try_fig("storm_variability_bz", lambda: plot_storm_variability(
            te_var_bz, ccf_var_bz, source_col="Bz", target_col="SYMH"))
    try:
        decouple_figs = plot_decoupling_summary(lag_df)
        for drv, fig in decouple_figs.items():
            figs[f"decoupling_{drv.lower()}"] = fig
    except Exception as e:
        log.error("Decoupling summary figures failed (%s); skipping.", e)

    for name, fig in figs.items():
        try:
            path = os.path.join(out, f"fig_{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            log.info("  Saved: %s", path)
        except Exception as e:
            log.error("Saving figure '%s' failed (%s); skipping.", name, e)

    # ── Summary CSVs ──────────────────────────────────────────────────────────
    lag_df.to_csv(os.path.join(out, "phase_lag_table.csv"), index=False)

    te_rows = []
    for (drv, rsp), res in te_results.items():
        te_rows.append({
            "type": "TE", "driver": drv, "response": rsp,
            "peak_lag_min": res["peak_lag_min"],
            "peak_te_bits": round(res["peak_te"], 5),
            "peak_p_raw":   round(res["peak_p"], 4),
            "peak_p_display": _fmt_p(res["peak_p"], te_n_surrogates),
            "significant":  bool(res["significant_mask"][
                res["lags_steps"] == res["peak_lag_step"]
            ].any()),
            "reliable":     res.get("reliable", True),
            "n_samples":    res.get("n_samples"),
            "n_surrogates": res.get("n_surrogates"),
        })
    pd.DataFrame(te_rows).to_csv(os.path.join(out, "te_results.csv"), index=False)

    log.info("\nAll results saved to: %s", out)

    return {
        "lag_df":               lag_df,
        "storms":               storms,
        "storms_categorized":   storms_categorized,
        "magnitude_stats":      magnitude_stats,
        "driver_df":            driver_df,
        "driver_type_df":       driver_type_df,
        "ccf_by_driver_type":   ccf_by_driver,
        "sea_by_cat":           sea_by_cat,
        "interp_sensitivity":   interp_sensitivity,
        "sea":                  sea,
        "te_results":           te_results,
        "gc_results":           gc_results,
        "phase_report":         phase_report,
        "te_var_bz":            te_var_bz,
        "ccf_var_bz":           ccf_var_bz,
        "wc_res":               wc_res,
        "cgc_res":              cgc_res,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Set to "1min" to rerun on 1-min OMNIWeb data (e.g. for the reviewer's
    # requested resolution sensitivity check) or leave as "5min" for the
    # original submission. All lag windows/limits rescale automatically —
    # see the RESOLUTION-AGNOSTIC TIME CONSTANTS block near the top of file.
    DATA_CADENCE = "1min"   # or "1min"

    run_pipeline(
        filepath_others  = r"C:\Users\user1\Desktop\B_REP\1st  Research LAG SUBMITTED\omni_others_2014_2016.txt",
        filepath_kp      = r"C:\Users\user1\Desktop\B_REP\1st  Research LAG SUBMITTED\omni_kp_2014_2016.txt",
        output_dir       = r"C:\Users\user1\Desktop\B_REP\1st  Research LAG SUBMITTED\\results",
        target_freq      = DATA_CADENCE,
        te_n_surrogates  = 200,    # use 200–500 for final publication
        te_k             = 5,
        te_max_lag_steps = None,   # auto-derived from TE_MAX_LAG_MIN (240 min)
        te_n_max         = 30000,   
    )
