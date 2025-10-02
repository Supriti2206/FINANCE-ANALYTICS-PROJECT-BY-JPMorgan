from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from typing import List, Tuple

# plotting
import matplotlib.pyplot as plt


def load_csv(path: str) -> List[Tuple[int, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # case-insensitive column finding
        cols = {k.lower(): k for k in reader.fieldnames}
        fico_col = cols.get("fico_score") or cols.get("fico") or cols.get("fico_score")
        default_col = cols.get("default") or cols.get("is_default")
        if fico_col is None or default_col is None:
            raise ValueError("Could not find 'fico_score' and 'default' columns in CSV")
        for r in reader:
            try:
                fico = int(float(r[fico_col]))
                default = int(float(r[default_col]))
            except Exception:
                continue
            rows.append((fico, default))
    return rows


def aggregate_by_fico(rows: List[Tuple[int, int]]) -> Tuple[List[int], List[int], List[int]]:
    agg = defaultdict(lambda: [0, 0])
    for fico, d in rows:
        agg[fico][0] += 1
        agg[fico][1] += int(d)
    scores = sorted(agg.keys())
    n = [agg[s][0] for s in scores]
    k = [agg[s][1] for s in scores]
    return scores, n, k


def prefix_sums(arr: List[int]) -> List[int]:
    pref = [0]
    s = 0
    for x in arr:
        s += x
        pref.append(s)
    return pref


def build_cost_matrices(n: List[int], k: List[int]) -> Tuple[List[List[float]], List[List[float]]]:
    m = len(n)
    pref_n = prefix_sums(n)  # length m+1
    pref_k = prefix_sums(k)

    cost_mse = [[0.0] * m for _ in range(m)]
    cost_ll = [[0.0] * m for _ in range(m)]

    for i in range(m):
        for j in range(i, m):
            n_sum = pref_n[j + 1] - pref_n[i]
            k_sum = pref_k[j + 1] - pref_k[i]
            if n_sum == 0:
                cost_mse[i][j] = 0.0
                cost_ll[i][j] = 0.0
                continue
            p = k_sum / n_sum
            # MSE cost for binary labels: sum (y - p)^2 = n * p * (1-p)
            cost_mse[i][j] = n_sum * p * (1.0 - p)

            # Log-likelihood (Bernoulli): k ln p + (n-k) ln(1-p)
            # define 0*ln0 = 0
            ll = 0.0
            if k_sum > 0:
                ll += k_sum * math.log(p)
            if n_sum - k_sum > 0:
                ll += (n_sum - k_sum) * math.log(1.0 - p)
            cost_ll[i][j] = -ll

    return cost_mse, cost_ll


def dp_partition(cost: List[List[float]], m: int, r: int) -> List[int]:
    INF = float("inf")
    # dp[k][j] minimum cost to partition first j items (0..j-1) into k segments
    # We'll use 0..r and 0..m
    dp = [[INF] * (m + 1) for _ in range(r + 1)]
    prev = [[-1] * (m + 1) for _ in range(r + 1)]
    dp[0][0] = 0.0

    for kseg in range(1, r + 1):
        for j in range(1, m + 1):
            # choose i from 0..j-1 as previous cut
            best_cost = INF
            best_i = -1
            for i in range(0, j):
                c = dp[kseg - 1][i]
                if c == INF:
                    continue
                seg_cost = cost[i][j - 1]  # cost for i..j-1
                tot = c + seg_cost
                if tot < best_cost:
                    best_cost = tot
                    best_i = i
            dp[kseg][j] = best_cost
            prev[kseg][j] = best_i

    # Reconstruct starts
    starts = [0] * r
    cur_j = m
    for kseg in range(r, 0, -1):
        i = prev[kseg][cur_j]
        if i is None or i < 0:
            # fallback: evenly spaced partition
            starts = [int(round(x * m / r)) for x in range(r)]
            break
        starts[kseg - 1] = i
        cur_j = i
    # starts gives the starting index of each segment; ensure first is 0
    starts[0] = 0
    # If some starts are equal or non-increasing, fix to at least monotonic
    for idx in range(1, r):
        if starts[idx] <= starts[idx - 1]:
            starts[idx] = starts[idx - 1] + 1
            if starts[idx] >= m:
                starts[idx] = m - 1

    return starts


def starts_to_boundaries(starts: List[int], scores: List[int]) -> List[Tuple[int, int]]:
    m = len(scores)
    r = len(starts)
    bounds = []
    for i in range(r):
        lo_idx = starts[i]
        hi_idx = (starts[i + 1] - 1) if i + 1 < r else m - 1
        bounds.append((scores[lo_idx], scores[hi_idx]))
    return bounds


def compute_bucket_stats(bounds: List[Tuple[int, int]], scores: List[int], n: List[int], k: List[int]):
    res = []
    # build index map score->index
    idx = {s: i for i, s in enumerate(scores)}
    for lo, hi in bounds:
        i = idx[lo]
        j = idx[hi]
        n_sum = sum(n[i : j + 1])
        k_sum = sum(k[i : j + 1])
        p = k_sum / n_sum if n_sum > 0 else 0.0
        res.append({"lo": lo, "hi": hi, "n": n_sum, "k": k_sum, "p": p})
    return res


def mapping_from_bounds(bounds: List[Tuple[int, int]], scores: List[int]) -> dict:
    score_to_rating = {}
    ordered = bounds[:]  # already in ascending
    r = len(ordered)
    for i, (lo, hi) in enumerate(ordered):
        rating = r - i
        for s in range(lo, hi + 1):
            score_to_rating[s] = rating
    return score_to_rating


def write_mapping_csv(path: str, scores: List[int], map_mse: dict, map_ll: dict, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fico", "rating_mse", "rating_ll"])
        for s in scores:
            w.writerow([s, map_mse.get(s, ""), map_ll.get(s, "")])


def plot_pd_buckets(stats_mse: List[dict], stats_ll: List[dict], r: int, out_path: str):
    pd_mse = [b["p"] for b in stats_mse]
    pd_ll = [b["p"] for b in stats_ll]
    x = list(range(1, r + 1))

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

    # common style
    for ax, pd_vals, label in ((axes[0], pd_mse, f"K={r} (MSE)"), (axes[1], pd_ll, f"K={r} (LL)")):
        bars = ax.bar(x, pd_vals, color="#f39c12", edgecolor="black")
        ax.set_xlim(0.5, r + 0.5)
        ax.set_xticks(x)
        ax.set_xlabel("bucket index (low FICO -> high FICO)")
        ax.set_ylabel("PD (defaults / n)")
        ax.set_title(f"Estimated PD by bucket - {label}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # add value labels on top of bars
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)



def print_bucket_table(name: str, stats: List[dict]):
    print(f"\n{name} buckets (best = rating 1 -> highest FICO):")
    print("#\tFICO_lo-FICO_hi\tn\tk\tp")
    for i, b in enumerate(stats, 1):
        print(f"{i}\t{b['lo']}-{b['hi']}\t{b['n']}\t{b['k']}\t{b['p']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="Task 3 and 4_Loan_Data.csv")
    p.add_argument("--buckets", type=int, default=5)
    p.add_argument("--out", default="fico_rating_map.csv")
    p.add_argument("--plot-out", default="fico_pd_buckets.png")
    args = p.parse_args()

    rows = load_csv(args.csv)
    scores, n, k = aggregate_by_fico(rows)
    print(f"Loaded {sum(n)} records, {len(scores)} unique FICO scores. Buckets={args.buckets}")

    cost_mse, cost_ll = build_cost_matrices(n, k)

    # Partition
    m = len(scores)
    r = args.buckets
    starts_mse = dp_partition(cost_mse, m, r)
    starts_ll = dp_partition(cost_ll, m, r)

    bounds_mse = starts_to_boundaries(starts_mse, scores)
    bounds_ll = starts_to_boundaries(starts_ll, scores)

    stats_mse = compute_bucket_stats(bounds_mse, scores, n, k)
    stats_ll = compute_bucket_stats(bounds_ll, scores, n, k)

    print_bucket_table("MSE", stats_mse)
    print_bucket_table("Log-likelihood", stats_ll)

    map_mse = mapping_from_bounds(bounds_mse, scores)
    map_ll = mapping_from_bounds(bounds_ll, scores)

    write_mapping_csv(args.csv, scores, map_mse, map_ll, args.out)
    print(f"\nWrote mapping for {len(scores)} FICO values to {args.out}")
    # create plot similar to the attached example
    plot_pd_buckets(stats_mse, stats_ll, r, args.plot_out)
    print(f"Saved PD-by-bucket plot to {args.plot_out}")


if __name__ == "__main__":
    main()
