import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family":"serif","font.size":9,
                     "axes.linewidth":0.8,"figure.dpi":300})

r = json.load(open("data/volume_control_results.json"))
sira = ["A","C","B"]
ad = ["Version A\n347 seq, 7 cities",
      "Version C\n347 seq, 13 cities",
      "Version B\n793 seq, 13 cities"]

fig, ax = plt.subplots(figsize=(5.0, 3.4))
for i, k in enumerate(sira):
    v = r[k]["losses"]
    ax.scatter([i]*len(v), v, s=26, facecolors="none",
               edgecolors="#1f4e79", linewidths=1.0, zorder=3)
    m, sd = np.mean(v), np.std(v)
    ax.errorbar(i, m, yerr=sd, fmt="_", color="#c00000",
                markersize=22, capsize=5, lw=1.6, zorder=4)

ax.set_xticks(range(3)); ax.set_xticklabels(ad, fontsize=8)
ax.set_ylabel("Validation loss (MSE)")
ax.set_xlim(-0.5, 2.5)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.set_title("Volume-controlled geographic ablation (5 seeds each)",
             fontsize=9, pad=8)
fig.tight_layout()
fig.savefig("figures/figure_volume_control.pdf", bbox_inches="tight")
fig.savefig("figures/figure_volume_control.png", bbox_inches="tight")
print("Kaydedildi: figures/figure_volume_control.pdf / .png")
