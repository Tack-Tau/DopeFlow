# Check if K point has even more negative frequencies
from phonopy import load

phonon = load("phonopy_params.yaml")

# Check Gamma point (what you showed)
phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
qdict = phonon.get_qpoints_dict()
gamma_freqs = qdict["frequencies"][0]
print(f"Gamma point min freq: {gamma_freqs[0]:.4f} THz")

# Check K point
phonon.run_qpoints([[1/3, 1/3, 0.0]], with_eigenvectors=True)
qdict = phonon.get_qpoints_dict()
k_freqs = qdict["frequencies"][0]
print(f"K point min freq: {k_freqs[0]:.4f} THz")

# Which is more negative?
if k_freqs[0] < gamma_freqs[0]:
    print(f"\n=> K point has softer mode! Use K point for modulation.")
else:
    print(f"\n=> Gamma point has softer mode! Use Gamma point for modulation.")
