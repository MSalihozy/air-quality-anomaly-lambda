import json
import numpy as np

r = json.load(open("data/volume_control_results.json"))

print("=== TABLO: Hacim kontrollu ablasyon ===\n")
print(f"{'Config':<28}{'Seq':>6}{'Cities':>8}{'Mean':>9}{'SD':>8}{'Min':>8}{'Max':>8}")
print("-"*75)
etiket = {"A": ("Version A (Turkey excluded)", 7),
          "B": ("Version B (full coverage)", 13),
          "C": ("Version C (volume-matched)", 13)}
for k in ["A", "C", "B"]:
    d = r[k]; ad, nc = etiket[k]
    print(f"{ad:<28}{d['n']:>6}{nc:>8}{d['mean']:>9.3f}"
          f"{d['std']:>8.3f}{d['min']:>8.3f}{d['max']:>8.3f}")

print("\n=== Seed bazinda eslestirme (A vs B) ===")
A, B = r["A"]["losses"], r["B"]["losses"]
for i, s in enumerate([42,43,44,45,46]):
    kazanan = "B" if B[i] < A[i] else "A"
    print(f"  seed {s}:  A={A[i]:.3f}  B={B[i]:.3f}   -> {kazanan}")
print(f"\nB'nin kazandigi: {sum(1 for i in range(5) if B[i]<A[i])}/5")
print(f"Fark (A-B)     : {np.mean(A)-np.mean(B):+.3f}")
print(f"Havuzlanmis SD : {np.sqrt((np.var(A)+np.var(B))/2):.3f}")
