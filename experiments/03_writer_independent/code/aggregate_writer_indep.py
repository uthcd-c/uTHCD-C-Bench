#!/usr/bin/env python3
"""
Aggregate the writer-independent experiment and contrast it with the writer-
DEPENDENT 70:30 baseline (outputs_split/ratio_70, same pipeline, same ratio --
the ONLY difference is writer-disjointness).

For every model:
  * WI clean Top-1  : mean +/- std over seeds {1,42,123}
  * WI mCE@top-1    : mean +/- std over seeds
  * WD clean Top-1  : ratio_70 (writer-dependent, seed 1)
  * WD mCE@top-1    : ratio_70
  * writer gap      : WD_clean - WI_clean  (drop from unseen writers)

Then the robustness RANKING (by mCE@top-1) is compared WI-vs-WD with Kendall tau
and Spearman rho (per WI seed and for the WI mean), to test whether the paper's
ordering survives a writer-independent protocol.

Outputs:
  outputs_writer/writer_indep_summary.csv     (per-model table)
  outputs_writer/writer_indep_ranking.csv     (WD vs WI mean ranks + tau/rho)
  prints a console summary + per-corruption collapse table for fragile models.
"""
import json, csv, statistics as st
from pathlib import Path
from itertools import combinations

SEEDS = [1, 42, 123]
WI = Path("outputs_writer")
WD = Path("outputs_split/ratio_70")
MODELS = ["vgg16_bn", "googlenet", "swin_t", "efficientnet_b0", "squeezenet1_0",
          "convnext_tiny", "regnet_x_400mf", "shufflenet_v2_x0_5", "mnasnet0_5"]
CORR = ["gaussian_noise", "shot_noise", "impulse_noise", "gaussian_blur",
        "defocus_blur", "stroke_thinning", "elastic", "pixelate", "contrast", "scale"]


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def kendall_tau(a, b):
    n = len(a); c = d = 0
    for i, j in combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c - d) / (c + d) if (c + d) else 0.0


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for rank, i in enumerate(order): r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    dd = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * dd / (n * (n * n - 1))


def main():
    # ---- gather per-model metrics ----
    wd_clean, wd_mce = {}, {}
    for m in MODELS:
        cr = load(WD / m / "corruption_results_A.json")
        wd_clean[m] = cr["clean"]["top1"] if cr else None
        wd_mce[m] = cr["mce"]["top1"] if cr else None

    wi_clean = {m: [] for m in MODELS}
    wi_mce = {m: [] for m in MODELS}
    wi_mce_by_seed = {s: {} for s in SEEDS}
    for s in SEEDS:
        for m in MODELS:
            cr = load(WI / f"seed_{s}" / m / "corruption_results_A.json")
            if cr:
                wi_clean[m].append(cr["clean"]["top1"])
                wi_mce[m].append(cr["mce"]["top1"])
                wi_mce_by_seed[s][m] = cr["mce"]["top1"]

    def ms(v):
        if not v: return (float("nan"), float("nan"))
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)

    # ---- per-model summary table ----
    rows = []
    print("\n================  WRITER-INDEPENDENT vs WRITER-DEPENDENT (70:30)  ================")
    print(f"{'model':20s} {'WD_clean':>8s} {'WI_clean(mean+-std)':>20s} {'gap':>6s} "
          f"{'WD_mCE@1':>9s} {'WI_mCE@1(mean+-std)':>20s}")
    for m in MODELS:
        cm, cs = ms(wi_clean[m]); mm, msd = ms(wi_mce[m])
        gap = (wd_clean[m] - cm) if wd_clean[m] is not None else float("nan")
        print(f"{m:20s} {wd_clean[m]:8.2f} {cm:13.2f} +-{cs:4.2f} {gap:6.2f} "
              f"{wd_mce[m]:9.3f} {mm:13.3f} +-{msd:4.3f}")
        rows.append({"model": m, "WD_clean_top1": round(wd_clean[m], 3),
                     "WI_clean_top1_mean": round(cm, 3), "WI_clean_top1_std": round(cs, 3),
                     "writer_gap": round(gap, 3),
                     "WD_mCE_top1": round(wd_mce[m], 4),
                     "WI_mCE_top1_mean": round(mm, 4), "WI_mCE_top1_std": round(msd, 4),
                     **{f"WI_mCE_seed{s}": round(wi_mce_by_seed[s].get(m, float('nan')), 4) for s in SEEDS}})

    # overall (mean over models)
    all_wd_c = [wd_clean[m] for m in MODELS]
    all_wi_c = [ms(wi_clean[m])[0] for m in MODELS]
    print("-" * 88)
    print(f"{'MEAN over models':20s} {st.mean(all_wd_c):8.2f} {st.mean(all_wi_c):13.2f} "
          f"{'':6s} {st.mean(all_wd_c)-st.mean(all_wi_c):6.2f}")

    WI.mkdir(exist_ok=True)
    with open(WI / "writer_indep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- ranking consistency (by mCE@top-1, ascending = more robust) ----
    wd_order = [wd_mce[m] for m in MODELS]
    wi_mean_order = [ms(wi_mce[m])[0] for m in MODELS]
    print("\n================  ROBUSTNESS RANKING CONSISTENCY (mCE@top-1)  ================")
    tau_m = kendall_tau(wd_order, wi_mean_order); rho_m = spearman(wd_order, wi_mean_order)
    print(f"WD(writer-dep)  vs  WI-mean : Kendall tau={tau_m:.3f}  Spearman rho={rho_m:.3f}")
    rank_rows = [{"pair": "WD_vs_WImean", "kendall_tau": round(tau_m, 3), "spearman_rho": round(rho_m, 3)}]
    for s in SEEDS:
        if len(wi_mce_by_seed[s]) == len(MODELS):
            so = [wi_mce_by_seed[s][m] for m in MODELS]
            t = kendall_tau(wd_order, so); r = spearman(wd_order, so)
            print(f"WD(writer-dep)  vs  WI-seed{s:<3d}: Kendall tau={t:.3f}  Spearman rho={r:.3f}")
            rank_rows.append({"pair": f"WD_vs_WIseed{s}", "kendall_tau": round(t, 3), "spearman_rho": round(r, 3)})

    order_names = sorted(MODELS, key=lambda m: ms(wi_mce[m])[0])
    print("\nWI consensus robustness order (most->least robust by mean mCE@top-1):")
    for i, m in enumerate(order_names, 1):
        print(f"  {i:2d}. {m:20s} mCE@1={ms(wi_mce[m])[0]:.3f}")
    with open(WI / "writer_indep_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "kendall_tau", "spearman_rho"])
        w.writeheader(); w.writerows(rank_rows)

    # ---- collapse persistence for fragile models (gaussian_noise sev1 top-1 err) ----
    print("\n================  MILD-CORRUPTION COLLAPSE CHECK (gaussian_noise s1, top-1 err%)  ===========")
    print(f"{'model':20s} {'WD_err':>7s} {'WI_err(mean)':>12s}")
    for m in ["mnasnet0_5", "shufflenet_v2_x0_5", "convnext_tiny", "swin_t", "vgg16_bn"]:
        crwd = load(WD / m / "corruption_results_A.json")
        wde = crwd["corrupted"]["gaussian_noise"]["1"]["top1"] if crwd else float("nan")
        wies = []
        for s in SEEDS:
            crwi = load(WI / f"seed_{s}" / m / "corruption_results_A.json")
            if crwi: wies.append(crwi["corrupted"]["gaussian_noise"]["1"]["top1"])
        print(f"{m:20s} {wde:7.2f} {st.mean(wies) if wies else float('nan'):12.2f}")

    print(f"\nsaved {WI/'writer_indep_summary.csv'} and {WI/'writer_indep_ranking.csv'}")


if __name__ == "__main__":
    main()
