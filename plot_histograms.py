"""
plot_histograms.py

Histograms for an arbitrary number of lightning datasets:
  - daily counts over the period
  - hourly counts per day (grid)
  - diurnal cycle aggregated over the period

API uses a dict {label: dataframe} so adding a third/fourth source
(e.g. LINET flashes, LI corrected) is trivial.

Example:
    from data_io import read_strokes_dir, DOMAIN
    from plot_histograms import plot_all

    sources = {
        "LINET strokes":  read_strokes_dir("linet_data",       pattern="LinetStrokesData*.txt",  domain=DOMAIN),
        "LINET flashes":  read_strokes_dir("linet_data",       pattern="LinetFlashesData*.txt", domain=DOMAIN),
        "LI uncorrected": read_strokes_dir("LI_september/daily", pattern="*_uncorrected*.txt",  domain=DOMAIN),
    }
    plot_all(sources, outdir="plots/")
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default colour palette — index = order of insertion in the sources dict.
# Two LINET-related entries share a blue family; LI keeps the orange.
DEFAULT_COLORS = [
    "#1e3a8a",  # dark blue   — LINET strokes
    "#60a5fa",  # light blue  — LINET flashes
    "#d97706",  # orange      — LI uncorrected
    "#7c3aed",  # purple      — LI corrected (reserved for later)
    "#15803d",  # green       — reserved
]


# -------- internal helpers --------

def _counts_per_day(df):
    return df.groupby(df["datetime"].dt.date).size()


def _counts_per_hour(df):
    g = df.groupby([df["datetime"].dt.date, df["datetime"].dt.hour]).size()
    g.index.names = ["date", "hour"]
    return g


def _counts_diurnal(df):
    return df.groupby(df["datetime"].dt.hour).size().reindex(range(24), fill_value=0)


def _colors_for(sources, colors=None):
    """Returns a list of colors of the same length as sources dict."""
    if colors is None:
        colors = DEFAULT_COLORS
    return [colors[i % len(colors)] for i in range(len(sources))]


def _bar_offsets(n, total_width=0.8):
    """Compute offsets and bar width for n side-by-side bars centred on each x tick."""
    width = total_width / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    return offsets, width


# -------- plotting functions --------

def plot_daily(sources, outpath, log=False, colors=None,
               title="Daily stroke/flash counts, domain 6–18°E / 37–47°N"):
    """
    Daily counts for each source in `sources` (dict {label: DataFrame}).
    """
    cols = _colors_for(sources, colors)

    daily = {name: _counts_per_day(df) for name, df in sources.items()}
    all_dates = sorted(set().union(*[s.index for s in daily.values()]))
    for name in daily:
        daily[name] = daily[name].reindex(all_dates, fill_value=0)

    years_in_data = {d.year for d in all_dates}
    fmt = "%Y-%m-%d" if len(years_in_data) > 1 else "%m-%d"

    x = np.arange(len(all_dates))
    offsets, width = _bar_offsets(len(sources))

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(all_dates)), 5))
    for (name, counts), off, col in zip(daily.items(), offsets, cols):
        ax.bar(x + off, counts.values, width, label=name, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime(fmt) for d in all_dates], rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Counts" + (" (log)" if log else ""))
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()
    print(f"  saved {outpath}")


def plot_hourly_per_day(sources, outpath, ncols=5, colors=None):
    """
    Grid of per-day hourly histograms, one subplot per date.
    Bars from all sources are placed side-by-side within each hour.
    """
    cols = _colors_for(sources, colors)

    hourly = {name: _counts_per_hour(df) for name, df in sources.items()}
    all_dates = sorted(set().union(*[set(d for d, _ in h.index) for h in hourly.values()]))
    nrows = int(np.ceil(len(all_dates) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows),
                             sharey=False, squeeze=False)

    x = np.arange(24)
    offsets, width = _bar_offsets(len(sources))

    for ax, date in zip(axes.flat, all_dates):
        for (name, h_series), off, col in zip(hourly.items(), offsets, cols):
            vals = np.array([h_series.get((date, h), 0) for h in range(24)])
            ax.bar(x + off, vals, width, color=col, label=name)
        ax.set_title(date.strftime("%Y-%m-%d"), fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.flat[len(all_dates):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sources),
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Hourly counts per day", fontsize=12, y=1.02)
    fig.supxlabel("Hour (UTC)", fontsize=10)
    fig.supylabel("Counts", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {outpath}")


def plot_diurnal(sources, outpath, normalize=False, colors=None):
    """
    Aggregated diurnal cycle for each source.
    With normalize=True, each curve is divided by its own total — best for
    comparing shape (timezone check, peak timing).
    """
    cols = _colors_for(sources, colors)

    diurnal = {name: _counts_diurnal(df).astype(float) for name, df in sources.items()}
    if normalize:
        for name in diurnal:
            tot = diurnal[name].sum()
            if tot > 0:
                diurnal[name] = diurnal[name] / tot
        ylabel = "Fraction"
    else:
        ylabel = "Counts (total over period)"

    x = np.arange(24)
    offsets, width = _bar_offsets(len(sources))

    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, d), off, col in zip(diurnal.items(), offsets, cols):
        ax.bar(x + off, d.values, width, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title("Diurnal cycle" + (", normalized" if normalize else ""))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()
    print(f"  saved {outpath}")


def print_summary_table(sources):
    """Short totals + top days for each source. Useful sanity check in the notebook."""
    print("\n--- summary ---")
    for name, df in sources.items():
        print(f"  {name:20s}: {len(df):>10}")

    # Pairwise ratios — handy when comparing strokes vs flashes vs LI groups
    names = list(sources.keys())
    if len(names) >= 2:
        print("\n  ratios (rows / cols):")
        totals = {n: len(df) for n, df in sources.items()}
        header = "  " + " " * 22 + "".join(f"{n:>16s}" for n in names)
        print(header)
        for ni in names:
            line = f"  {ni:22s}"
            for nj in names:
                if totals[nj] == 0:
                    line += f"{'—':>16s}"
                else:
                    line += f"{totals[ni] / totals[nj]:>15.2f}×"
            print(line)


def plot_all(sources, outdir="plots", colors=None):
    """Convenience: build the full set of plots."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {outdir}/")
    plot_daily(sources,  outdir / "daily_counts.png",     log=False, colors=colors)
    plot_daily(sources,  outdir / "daily_counts_log.png", log=True,  colors=colors)
    plot_hourly_per_day(sources, outdir / "hourly_per_day.png", colors=colors)
    plot_diurnal(sources, outdir / "diurnal_raw.png",        normalize=False, colors=colors)
    plot_diurnal(sources, outdir / "diurnal_normalized.png", normalize=True,  colors=colors)
    print_summary_table(sources)
