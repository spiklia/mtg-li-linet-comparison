"""
match_strokes.py

Matching between two lightning datasets (e.g. LINET flashes ↔ LI groups).
Both inputs are generic — any DataFrame with [lon, lat, datetime] columns works,
so the same code handles strokes-vs-LI, flashes-vs-LI, or any future pair.

Core function:
    matches = match_nearest(df_a, df_b, dt_max_ms=200, dr_max_km=20)

Returns one row per event in df_a:
    - idx_a:   original index in df_a
    - lon_a, lat_a, time_a
    - idx_b:   index of nearest match in df_b within window (NaN if no match)
    - lon_b, lat_b, time_b (NaN if no match)
    - dt_ms:   signed time offset, b - a (NaN if no match)
    - dr_km:   spatial distance (NaN if no match)
    - matched: boolean
"""

import numpy as np
import pandas as pd


# -------- distance --------

def haversine_km(lon1, lat1, lon2, lat2):
    """
    Great-circle distance in km. All inputs may be scalars or numpy arrays.
    """
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# -------- matching --------

def match_nearest(df_a, df_b, dt_max_ms=200, dr_max_km=20.0):
    """
    For each event in df_a, find the spatially nearest event in df_b
    that falls inside the time window ±dt_max_ms.

    Both DataFrames must have columns 'lon', 'lat', 'datetime'.
    The original index of each input is preserved as 'idx_a' / 'idx_b'.

    Returns
    -------
    pd.DataFrame, one row per df_a event.
    """
    if len(df_a) == 0:
        return pd.DataFrame()

    # Sort df_b by time once — that's what we'll searchsort into
    b = df_b.sort_values("datetime")
    b_times = b["datetime"].values            # datetime64[ns]
    b_lon   = b["lon"].values
    b_lat   = b["lat"].values
    b_idx   = b.index.values                  # original index in df_b

    a_times = df_a["datetime"].values
    a_lon   = df_a["lon"].values
    a_lat   = df_a["lat"].values
    a_idx   = df_a.index.values

    dt_max = np.timedelta64(int(dt_max_ms * 1_000_000), "ns")

    # Pre-allocate result arrays
    n = len(df_a)
    out_idx_b  = np.full(n, -1,    dtype=np.int64)
    out_dt_ms  = np.full(n, np.nan)
    out_dr_km  = np.full(n, np.nan)
    out_lon_b  = np.full(n, np.nan)
    out_lat_b  = np.full(n, np.nan)
    out_time_b = np.empty(n, dtype="datetime64[ns]")
    out_time_b[:] = np.datetime64("NaT")

    # Searchsorted on the whole arrays at once — vectorized
    lo = np.searchsorted(b_times, a_times - dt_max, side="left")
    hi = np.searchsorted(b_times, a_times + dt_max, side="right")

    # Per-event candidate loop. Each window is tiny (handful of B events
    # for typical dt_max=200ms), so the loop is short and the distance
    # computation is vectorized inside.
    for i in range(n):
        if lo[i] == hi[i]:
            continue
        cand_lon = b_lon[lo[i]:hi[i]]
        cand_lat = b_lat[lo[i]:hi[i]]
        dr = haversine_km(a_lon[i], a_lat[i], cand_lon, cand_lat)
        in_range = dr <= dr_max_km
        if not in_range.any():
            continue
        # Choose the spatially nearest in-range candidate
        masked_dr = np.where(in_range, dr, np.inf)
        best_local = int(np.argmin(masked_dr))
        best_global = lo[i] + best_local

        out_idx_b[i]  = b_idx[best_global]
        out_dr_km[i]  = dr[best_local]
        out_lon_b[i]  = b_lon[best_global]
        out_lat_b[i]  = b_lat[best_global]
        out_time_b[i] = b_times[best_global]
        out_dt_ms[i]  = (b_times[best_global] - a_times[i]) / np.timedelta64(1, "ms")

    matched = out_idx_b != -1
    return pd.DataFrame({
        "idx_a":   a_idx,
        "lon_a":   a_lon,
        "lat_a":   a_lat,
        "time_a":  a_times,
        "idx_b":   np.where(matched, out_idx_b, np.nan),
        "lon_b":   out_lon_b,
        "lat_b":   out_lat_b,
        "time_b":  out_time_b,
        "dt_ms":   out_dt_ms,
        "dr_km":   out_dr_km,
        "matched": matched,
    })


# -------- diagnostics --------

def matching_summary(matches, label_a="A", label_b="B"):
    """Print key matching metrics."""
    n = len(matches)
    m = int(matches["matched"].sum())
    if n == 0:
        print("(empty)")
        return
    print(f"  {label_a} events:        {n}")
    print(f"  with {label_b} partner:  {m}  ({100*m/n:.1f}%)")
    print(f"  no partner:           {n - m}  ({100*(n-m)/n:.1f}%)")
    if m > 0:
        good = matches.loc[matches["matched"]]
        print(f"  |dt| median:           {good['dt_ms'].abs().median():.1f} ms")
        print(f"  |dt| 90th percentile:  {good['dt_ms'].abs().quantile(0.9):.1f} ms")
        print(f"  dr  median:            {good['dr_km'].median():.2f} km")
        print(f"  dr  90th percentile:   {good['dr_km'].quantile(0.9):.2f} km")
        # Signed offsets — non-zero median dt or systematic dr signs hint at bias
        print(f"  dt  median (signed):   {good['dt_ms'].median():+.1f} ms")
        print(f"  Δlon median (b−a):     {(good['lon_b'] - good['lon_a']).median():+.3f}°")
        print(f"  Δlat median (b−a):     {(good['lat_b'] - good['lat_a']).median():+.3f}°")

def matching_summary_table(matches_dict):
    """
    Comparison table of matching metrics for one or more matches.

    Parameters
    ----------
    matches_dict : {label: DataFrame}
        Output of match_nearest for one or more comparisons.

    Returns
    -------
    pd.DataFrame, metrics as rows, comparison cases as columns.
    """
    rows = {}
    for label, m in matches_dict.items():
        if len(m) == 0:
            continue
        good = m.loc[m["matched"]]
        rows[label] = {
            "Total events":          len(m),
            "Matched events":        int(m["matched"].sum()),
            "DE (%)":                100 * m["matched"].mean(),
            "|dt| median (ms)":      good["dt_ms"].abs().median(),
            "|dt| 90th pct (ms)":    good["dt_ms"].abs().quantile(0.9),
            "dr median (km)":        good["dr_km"].median(),
            "dr 90th pct (km)":      good["dr_km"].quantile(0.9),
            "dt signed median (ms)": good["dt_ms"].median(),
            "Δlon median (°)":       (good["lon_b"] - good["lon_a"]).median(),
            "Δlat median (°)":       (good["lat_b"] - good["lat_a"]).median(),
        }
    return pd.DataFrame(rows)

def plot_match_diagnostics(matches, outpath=None, title=""):
    """
    Two-panel figure: distributions of dt and dr for matched pairs.
    Useful to see whether your chosen window is too tight or too loose.
    """
    import matplotlib.pyplot as plt
    good = matches.loc[matches["matched"]]
    if len(good) == 0:
        print("no matched pairs — skip diagnostics")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(good["dt_ms"], bins=60, color="#1e3a8a", alpha=0.85)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("dt (ms) = time_b − time_a")
    axes[0].set_ylabel("Pairs")
    axes[0].set_title("Time offset")
    axes[0].grid(alpha=0.3)

    axes[1].hist(good["dr_km"], bins=60, color="#d97706", alpha=0.85)
    axes[1].set_xlabel("dr (km) = haversine(a, b)")
    axes[1].set_ylabel("Pairs")
    axes[1].set_title("Spatial distance")
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=130)
        print(f"  saved {outpath}")
    plt.show()


def sweep_thresholds(df_a, df_b, dt_list_ms, dr_list_km, verbose=False):
    """
    Run matching for a grid of (dt_max, dr_max) and return DE for each combination.

    Returns
    -------
    pd.DataFrame indexed by dt_max_ms, columns dr_max_km, values = match fraction.
    """
    out = pd.DataFrame(index=dt_list_ms, columns=dr_list_km, dtype=float)
    n_total = len(dt_list_ms) * len(dr_list_km)
    k = 0
    for dt in dt_list_ms:
        for dr in dr_list_km:
            k += 1
            if verbose:
                print(f"  [{k}/{n_total}] dt={dt} ms, dr={dr} km ...", end=" ", flush=True)
            m = match_nearest(df_a, df_b, dt_max_ms=dt, dr_max_km=dr)
            out.loc[dt, dr] = m["matched"].mean() if len(m) else np.nan
            if verbose:
                print(f"DE={out.loc[dt, dr]:.3f}")
    out.index.name = "dt_max_ms"
    out.columns.name = "dr_max_km"
    return out


def plot_detection_curves(sweeps_dict, outpath=None,
                          dt_lines_s=(0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
                          dr_lines_km=(1, 5, 10, 20, 30, 50, 70, 100)):
    """
    Multi-row figure showing detection efficiency as 1D curves:
    left panel  — DE vs match distance, one curve per dt threshold
    right panel — DE vs match time,     one curve per dr threshold

    Parameters
    ----------
    sweeps_dict : {label: sweep_df}
        Each sweep_df is output of `sweep_thresholds`. Index is dt_max_ms,
        columns are dr_max_km, values are match fractions.
    dt_lines_s, dr_lines_km : iterables
        Subset of grid values to show as separate lines. Must be present in
        the sweep index / columns (multiply dt_lines_s by 1000 for ms).
    """
    import matplotlib.pyplot as plt

    n_rows = len(sweeps_dict)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), squeeze=False)

    cmap = plt.get_cmap("tab10")

    for row_i, (label, sweep) in enumerate(sweeps_dict.items()):
        # ---- Left panel: DE vs distance, lines parameterised by dt ----
        ax_l = axes[row_i, 0]
        for j, dt_s in enumerate(dt_lines_s):
            dt_ms = int(dt_s * 1000)
            if dt_ms not in sweep.index:
                continue
            ax_l.plot(sweep.columns, sweep.loc[dt_ms].values * 100,
                      marker="x", color=cmap(j % 10), label=f"{dt_s} s")
        ax_l.set_xlabel("Match distance [km]")
        ax_l.set_ylabel("Rel. detection [%]")
        ax_l.set_title(f"({chr(97 + 2*row_i)})  {label}")
        ax_l.legend(ncol=2, fontsize=8, loc="lower right")
        ax_l.grid(alpha=0.3)

        # ---- Right panel: DE vs time, lines parameterised by dr ----
        ax_r = axes[row_i, 1]
        for j, dr_km in enumerate(dr_lines_km):
            if dr_km not in sweep.columns:
                continue
            ax_r.plot(np.array(sweep.index) / 1000.0,
                      sweep[dr_km].values * 100,
                      marker="+", color=cmap(j % 10), label=f"{dr_km} km")
        ax_r.set_xlabel("Match time [s]")
        ax_r.set_ylabel("Rel. detection [%]")
        ax_r.set_title(f"({chr(98 + 2*row_i)})  {label}")
        ax_r.legend(ncol=2, fontsize=8, loc="lower right")
        ax_r.grid(alpha=0.3)

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=130, bbox_inches="tight")
        print(f"  saved {outpath}")
    plt.show()
