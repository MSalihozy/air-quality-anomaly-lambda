import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8, "legend.frameon": True,
    "legend.fontsize": 8, "figure.dpi": 300,
})

A = json.load(open("data/models/lstm_metadata.json"))
B = json.load(open("data/models/lstm_metadata_full.json"))

def panel(ax, m, baslik):
    tr, va = m["history_train_loss"], m["history_val_loss"]
    ep = range(1, len(tr) + 1)
    best = min(range(len(va)), key=lambda i: va[i]) + 1

    ax.plot(ep, tr, "-",  color="#1f4e79", lw=1.4, label="Training Loss")
    ax.plot(ep, va, "--", color="#c00000", lw=1.4, label="Validation Loss")
    ax.axvline(best, color="grey", ls=":", lw=0.9)

    ax.set_title(baslik, fontsize=9, pad=6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_xlim(0, len(tr) + 1)
    ax.set_ylim(0, max(max(tr), max(va)) * 1.45)
    ax.legend(loc="upper right")

    bilgi = (f"Best epoch {best}, stopped at {len(tr)}\n"
             f"train = {tr[-1]:.3f}\n"
             f"val   = {va[-1]:.3f}")
    ax.text(0.97, 0.68, bilgi, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, linespacing=1.4,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="grey", lw=0.5))

fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.2))
panel(axes[0], A, f"(a) Version A — 7 cities, {A['n_train_sequences']} sequences")
panel(axes[1], B, f"(b) Version B — 13 cities, {B['n_train_sequences']} sequences")
fig.tight_layout()
fig.savefig("figures/figure2_lstm_curves.pdf", bbox_inches="tight")
fig.savefig("figures/figure2_lstm_curves.png", bbox_inches="tight")
print("Kaydedildi.\n")
def satir(ad, va, vb, fmt="{}"):
    print(f"{ad:<34} {fmt.format(va):>12} {fmt.format(vb):>12}")

print("=== TABLO 10 ===")
print(f"{'Metric':<34} {'Version A':>12} {'Version B':>12}")
satir("Active training cities", 7, 13)
satir("Training sequences (12-hour)", A["n_train_sequences"], B["n_train_sequences"])
satir("Validation sequences", A["n_val_sequences"], B["n_val_sequences"])
satir("Final training loss (MSE)", A["final_train_loss"], B["final_train_loss"], "{:.3f}")
satir("Final validation loss (MSE)", A["final_val_loss"], B["final_val_loss"], "{:.3f}")
satir("Train MSE p99 threshold", A["threshold_p99"], B["threshold_p99"], "{:.3f}")
satir("Training / validation ratio",
      A["final_val_loss"]/A["final_train_loss"],
      B["final_val_loss"]/B["final_train_loss"], "{:.2f}")
satir("Epochs completed", A["epochs_run"], B["epochs_run"])
