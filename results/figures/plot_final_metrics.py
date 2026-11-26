import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; C_M, C_S = "#eb6834", "#2a78d6"
panels = [("test-clean WER (%) $\\downarrow$", 45.08, 43.36, "{:.2f}"), ("FSD50K mAP (%) $\\uparrow$", 39.36, 43.07, "{:.2f}"),
          ("VoxCeleb1-O EER (%) $\\downarrow$", 8.44, 4.30, "{:.2f}"), ("Parameters (M) $\\downarrow$", 64.35, 22.44, "{:.2f}"),
          ("CPU real-time factor $\\downarrow$", 0.0360, 0.0246, "{:.4f}")]
fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.1))
for a, (t, m, s, fmt) in zip(ax, panels):
    bars = a.bar([0, 1], [m, s], color=[C_M, C_S], width=0.62, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, (m, s)): a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    a.set_title(t, fontsize=7.6, color=INK, pad=6); a.set_xticks([0, 1]); a.set_xticklabels(["M (64M)", "S (22M)"], fontsize=7, color=INK2)
    a.set_ylim(0, max(m, s) * 1.22); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color("#b5b4ad")
    a.tick_params(axis="y", colors=INK2, labelsize=6.5, length=2)
fig.tight_layout(w_pad=0.8); fig.savefig("fig_final_metrics.png", dpi=300); print("saved")
