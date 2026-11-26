import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8})
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C_S, C_M = "#2a78d6", "#eb6834"      # validated categorical slots 1 (blue) and 2 (orange)

def rows(p): return list(csv.reader(open(p)))
s1 = [[float(x) for x in r] for r in rows("s_stage1.csv")]
s2 = [[float(x) for x in r] for r in rows("s_stage2.csv")]
h1 = list(csv.DictReader(open("e0_stage1.csv"))); h2 = list(csv.DictReader(open("e0_stage2.csv")))
S1 = dict(ep=[r[0] for r in s1], a=[r[1] for r in s1], b=[r[2] for r in s1], c=[r[3] for r in s1])
M1 = dict(ep=[float(r["epoch"]) for r in h1], a=[float(r["asr_val_ratio"]) for r in h1], b=[100*float(r["at_val_mAP"]) for r in h1], c=[100*float(r["sv_val_eer"]) for r in h1])
S2 = dict(ep=[r[0] for r in s2], a=[r[1] for r in s2], b=[r[2] for r in s2], c=[r[3] for r in s2])
M2 = dict(ep=[float(r["epoch"]) for r in h2], a=[100*float(r["asr_dev_wer"]) for r in h2], b=[100*float(r["at_val_mAP"]) for r in h2], c=[100*float(r["sv_val_eer"]) for r in h2])

fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.55), facecolor=SURF)
titles = [["(a) Stage 1: ASR distillation ratio $\\downarrow$", "(b) Stage 1: AT mAP (%) $\\uparrow$", "(c) Stage 1: SV EER (%) $\\downarrow$"],
          ["(d) Stage 2: ASR WER, dev-clean (%) $\\downarrow$", "(e) Stage 2: AT mAP (%) $\\uparrow$", "(f) Stage 2: SV EER (%) $\\downarrow$"]]
for i, (S, M) in enumerate([(S1, M1), (S2, M2)]):
    for j, k in enumerate("abc"):
        a = ax[i, j]
        a.plot(M["ep"], M[k], color=C_M, lw=1.5, marker="o", ms=2.6, label="Zipformer-M (64M, baseline)", zorder=3)
        a.plot(S["ep"], S[k], color=C_S, lw=1.5, marker="o", ms=2.6, label="Zipformer-S (22M, proposed)", zorder=4)
        a.set_title(titles[i][j], color=INK, loc="left", pad=3)
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color("#b5b4ad")
        a.tick_params(colors=INK2, length=2.5)
        a.set_xlabel("Stage-1 epoch" if i == 0 else "Stage-2 epoch", color=INK2, labelpad=1)
        if i == 1:
            a.axvline(1.5, color="#8a8984", lw=0.8, ls=(0, (3, 2)), zorder=1)
            if j == 0: a.annotate("encoder unfrozen", xy=(1.5, 0.9), xycoords=a.get_xaxis_transform(), xytext=(6.5, 0.9), textcoords=a.get_xaxis_transform(),
                                  va="center", ha="left", fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.6))
            a.set_xticks([1, 5, 10, 15])
        else:
            a.set_xticks([1, 10, 20, 30])
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005), labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.9, w_pad=1.0)
fig.savefig("fig_training_curves.png", dpi=300); fig.savefig("fig_training_curves.pdf")
print("saved")
