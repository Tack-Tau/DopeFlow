#!/Users/tonyspc/miniforge3/envs/pyxtal/bin/python3
"""
Phonon Post-Processing for Displaced Structures

Analyzes relaxed energies, identifies lowest energy structure,
and post-processes phonon calculations on the optimal structure.

Usage:
    python3 analyze_phon.py                    # Analyze energies only
    python3 analyze_phon.py --process-phonon   # Full phonon post-processing
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phonopy import load, Phonopy
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

# Greek letter mapping for LaTeX labels
GREEK_LETTERS = {
    'GAMMA': r'\Gamma',
    'DELTA': r'\Delta',
    'LAMBDA': r'\Lambda',
    'SIGMA': r'\Sigma',
}

def convert_label_to_latex(label):
    """Convert label to LaTeX format."""
    if label.startswith('\\'):
        return f"${label}$"
    
    if '_' in label:
        base, subscript = label.split('_', 1)
    else:
        base = label
        subscript = None
    
    if base in GREEK_LETTERS:
        latex_label = GREEK_LETTERS[base]
        if subscript:
            latex_label = f"{latex_label}_{{{subscript}}}"
        return f"${latex_label}$"
    else:
        if subscript:
            return f"${base}_{{{subscript}}}$"
        return label

def extract_forces_from_vasprun(vasprun_paths):
    """Extract forces from vasprun.xml files using pymatgen."""
    forces_list = []
    
    for vasprun_path in vasprun_paths:
        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            forces = vr.ionic_steps[-1]['forces']
            forces_list.append(forces)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {vasprun_path}: {e}")
    
    return np.array(forces_list)

def create_force_constants(structure, phon_dir, forces, supercell_matrix):
    """Create force constants using phonopy Python API."""
    # Convert pymatgen structure to phonopy
    phonopy_structure = get_phonopy_structure(structure)
    
    # Create Phonopy object (factor now auto-set to THz)
    phonon = Phonopy(
        phonopy_structure,
        supercell_matrix=supercell_matrix,
        primitive_matrix='auto'
    )
    
    # Generate displacements
    phonon.generate_displacements(distance=0.01)
    
    # Set forces
    phonon.forces = forces
    
    # Generate force constants
    phonon.produce_force_constants()
    
    # Save force constants
    phonon.save(filename=str(phon_dir / 'phonopy_params.yaml'), settings={'force_constants': True})
    
    return phonon

def load_phonopy_data(phon_dir):
    """Load phonopy calculation data."""
    phon_path = Path(phon_dir)
    
    yaml_files = ['phonopy_params.yaml', 'phonopy_disp.yaml', 'phonopy.yaml']
    phonopy_yaml = None
    
    for yaml_file in yaml_files:
        yaml_path = phon_path / yaml_file
        if yaml_path.exists():
            phonopy_yaml = yaml_path
            break
    
    if phonopy_yaml is None:
        raise FileNotFoundError(f"No phonopy yaml file found in {phon_dir}")
    
    print(f"  Loading phonopy data from: {phonopy_yaml.name}")
    phonon = load(str(phonopy_yaml))
    
    return phonon

def plot_band_structure(phonon, structure, output_path, title=None):
    """Plot phonon band structure with pymatgen k-path."""
    kpath = HighSymmKpath(structure)
    kpath_dict = kpath.kpath
    
    print(f"  K-path: {kpath_dict['path']}")
    
    # Build band paths (nested list structure)
    qpoints_for_phonopy = []
    labels_plot = []
    
    for branch in kpath_dict['path']:
        branch_qpoints = []
        labels_plot.extend(branch)
        
        for i in range(len(branch) - 1):
            start = np.array(kpath_dict['kpoints'][branch[i]])
            end = np.array(kpath_dict['kpoints'][branch[i+1]])
            
            # 51 points between each pair
            for j in range(51):
                frac = j / 50.0
                qpoint = start + frac * (end - start)
                branch_qpoints.append(qpoint)
        
        qpoints_for_phonopy.append(branch_qpoints)
    
    # Run phonopy band structure
    phonon.run_band_structure(qpoints_for_phonopy, is_band_connection=False)
    band_dict = phonon.get_band_structure_dict()
    
    # Flatten results
    all_qpoints = []
    all_frequencies = []
    for qpts, freqs in zip(band_dict['qpoints'], band_dict['frequencies']):
        all_qpoints.extend(qpts)
        all_frequencies.extend(freqs)
    
    all_qpoints = np.array(all_qpoints)
    frequencies = np.array(all_frequencies)
    
    # Calculate distances manually in reciprocal space
    rec_lattice = structure.lattice.reciprocal_lattice
    distances = [0.0]
    
    for i in range(1, len(all_qpoints)):
        q_prev_cart = rec_lattice.get_cartesian_coords(all_qpoints[i-1])
        q_curr_cart = rec_lattice.get_cartesian_coords(all_qpoints[i])
        delta = np.linalg.norm(q_curr_cart - q_prev_cart)
        distances.append(distances[-1] + delta)
    
    distances = np.array(distances)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each band
    n_bands = frequencies.shape[1]
    for band_idx in range(n_bands):
        ax.plot(distances, frequencies[:, band_idx], 'r-', linewidth=1.0)
    
    # Add high-symmetry point markers
    tick_positions = [distances[0]]
    tick_labels = [convert_label_to_latex(labels_plot[0])]
    
    current_idx = 0
    for i, band_segment in enumerate(bands):
        segment_length = (len(band_segment) - 1) * 51
        current_idx += segment_length
        
        if current_idx < len(distances):
            tick_positions.append(distances[current_idx])
            
            # Get label index
            label_idx = sum(len(seg) for seg in bands[:i+1])
            if label_idx < len(labels_plot):
                tick_labels.append(convert_label_to_latex(labels_plot[label_idx]))
    
    for pos in tick_positions:
        ax.axvline(x=pos, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_xlabel('Wave vector', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency (THz)', fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='y', labelsize=14)
    
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path.name}")
    
    # Check for imaginary frequencies
    min_freq = np.min(frequencies)
    if min_freq < -0.1:
        print(f"  WARNING: Imaginary frequencies detected (min: {min_freq:.3f} THz)")
        return False
    else:
        print(f"  All frequencies positive (min: {min_freq:.3f} THz)")
        return True

def plot_dos(phonon, structure, output_path, mesh=[20, 20, 20]):
    """Calculate and plot phonon DOS."""
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_total_dos()
    phonon.run_projected_dos()
    
    total_dos_dict = phonon.get_total_dos_dict()
    proj_dos_dict = phonon.get_projected_dos_dict()
    
    frequencies = total_dos_dict['frequency_points']
    total_dos = total_dos_dict['total_dos']
    
    # Group projected DOS by element
    primitive = phonon.primitive
    element_dos = {}
    
    for i in range(len(primitive.masses)):
        symbol = primitive.symbols[i]
        pdos = proj_dos_dict['projected_dos'][i]
        
        if symbol in element_dos:
            element_dos[symbol] += pdos
        else:
            element_dos[symbol] = pdos.copy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(frequencies, total_dos, 'k-', linewidth=2.0, label='Total', alpha=0.8)
    
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(element_dos)))
    for idx, (symbol, pdos) in enumerate(element_dos.items()):
        ax.plot(frequencies, pdos, linewidth=1.5, 
                label=symbol, alpha=0.8, color=colors[idx])
    
    ax.set_xlabel('Frequency (THz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('DOS', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xlim(left=min(frequencies))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path.name}")

def check_gamma_frequencies(phonon):
    """Check Gamma point frequencies."""
    phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    qdict = phonon.get_qpoints_dict()
    frequencies = qdict["frequencies"][0]
    
    print("\n  Gamma point frequencies (first 10 modes):")
    for i in range(min(10, len(frequencies))):
        status = " (SOFT MODE)" if frequencies[i] < -0.01 else ""
        print(f"    Mode {i+1:2d}: {frequencies[i]:10.6f} THz{status}")
    
    n_negative = sum(1 for f in frequencies if f < -0.01)
    
    if n_negative == 0:
        print(f"\n  SUCCESS: All soft modes eliminated!")
        return True
    else:
        print(f"\n  WARNING: Still have {n_negative} soft mode(s)")
        return False

def plot_combined_band_dos(phonon, structure, output_path, title=None, mesh=[20, 20, 20]):
    """Plot combined band structure and DOS with shared y-axis (ElectrideFlow style)."""
    # Set matplotlib rcParams to match ElectrideFlow
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Get k-path
    kpath = HighSymmKpath(structure)
    kpath_dict = kpath.kpath
    
    # Build q-points for band structure (nested list structure)
    qpoints_for_phonopy = []
    segment_markers = [0]
    
    for branch in kpath_dict['path']:
        branch_qpoints = []
        for i in range(len(branch) - 1):
            start = np.array(kpath_dict['kpoints'][branch[i]])
            end = np.array(kpath_dict['kpoints'][branch[i+1]])
            
            for j in range(51):
                frac = j / 50.0
                qpoint = start + frac * (end - start)
                branch_qpoints.append(qpoint)
        
        qpoints_for_phonopy.append(branch_qpoints)
        segment_markers.append(segment_markers[-1] + len(branch_qpoints))
    
    # Calculate band structure using run_band_structure
    phonon.run_band_structure(qpoints_for_phonopy, is_band_connection=False)
    band_dict = phonon.get_band_structure_dict()
    
    # Flatten results
    all_qpoints = []
    all_frequencies = []
    for qpts, freqs in zip(band_dict['qpoints'], band_dict['frequencies']):
        all_qpoints.extend(qpts)
        all_frequencies.extend(freqs)
    
    all_qpoints = np.array(all_qpoints)
    frequencies_band = np.array(all_frequencies)
    
    # Calculate distances manually in reciprocal space
    rec_lattice = structure.lattice.reciprocal_lattice
    distances_band = [0.0]
    
    for i in range(1, len(all_qpoints)):
        q_prev_cart = rec_lattice.get_cartesian_coords(all_qpoints[i-1])
        q_curr_cart = rec_lattice.get_cartesian_coords(all_qpoints[i])
        delta = np.linalg.norm(q_curr_cart - q_prev_cart)
        distances_band.append(distances_band[-1] + delta)
    
    distances_band = np.array(distances_band)
    
    # Calculate DOS
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_total_dos()
    phonon.run_projected_dos()
    
    total_dos_dict = phonon.get_total_dos_dict()
    proj_dos_dict = phonon.get_projected_dos_dict()
    
    frequencies_dos = total_dos_dict['frequency_points']
    total_dos = total_dos_dict['total_dos']
    
    # Get projected DOS by element
    primitive = phonon.primitive
    element_dos = {}
    for i in range(len(primitive.masses)):
        symbol = primitive.symbols[i]
        pdos = proj_dos_dict['projected_dos'][i]
        
        if symbol in element_dos:
            element_dos[symbol] += pdos
        else:
            element_dos[symbol] = pdos.copy()
    
    # Create combined plot (ElectrideFlow style: figsize=(12,6), wspace=0.08)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True,
                                   gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.08})
    
    # Plot band structure
    n_bands = frequencies_band.shape[1]
    for band_idx in range(n_bands):
        ax1.plot(distances_band, frequencies_band[:, band_idx], 
                'r-', linewidth=1.0, zorder=2)
    
    # Add segment markers
    tick_positions = []
    tick_labels = []
    
    current_pos = 0
    for branch_idx, branch in enumerate(kpath_dict['path']):
        for i, label in enumerate(branch):
            if i == 0 or (branch_idx > 0 and i == 0):
                # Start of segment
                tick_positions.append(distances_band[current_pos] if current_pos < len(distances_band) else distances_band[-1])
                tick_labels.append(convert_label_to_latex(label))
            elif i == len(branch) - 1:
                # End of segment
                idx = segment_markers[branch_idx + 1] - 1
                if idx < len(distances_band):
                    tick_positions.append(distances_band[idx])
                    tick_labels.append(convert_label_to_latex(label))
        
        if branch_idx < len(kpath_dict['path']) - 1:
            current_pos = segment_markers[branch_idx + 1]
    
    # Remove duplicate positions
    unique_ticks = []
    unique_labels = []
    for pos, label in zip(tick_positions, tick_labels):
        if not unique_ticks or abs(pos - unique_ticks[-1]) > 1e-6:
            unique_ticks.append(pos)
            unique_labels.append(label)
        else:
            # Merge labels at same position
            unique_labels[-1] = f"{unique_labels[-1]}|{label}"
    
    for pos in unique_ticks:
        ax1.axvline(x=pos, color='black', linestyle='-', linewidth=0.5, zorder=1)
    
    ax1.set_xticks(unique_ticks)
    ax1.set_xticklabels(unique_labels, fontsize=14)
    ax1.set_xlabel('Wave vector', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Frequency (THz)', fontsize=16, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5, zorder=5)
    ax1.set_xlim(distances_band[0] - 0.0001, distances_band[-1] + 0.0001)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    
    # Plot DOS (ElectrideFlow style: linewidth=1.0, arange colors)
    ax2.plot(total_dos, frequencies_dos, 'k-', linewidth=1.0, 
             label='Total', alpha=0.8, zorder=3)
    
    # Use arange for colors (ElectrideFlow style)
    if element_dos:
        colors = plt.cm.tab10(np.arange(0, 0.2 * len(element_dos), 0.2))
        for idx, (symbol, pdos) in enumerate(element_dos.items()):
            ax2.plot(pdos, frequencies_dos, linewidth=1.5, 
                    label=symbol, alpha=0.8, color=colors[idx], zorder=2)
    
    ax2.set_xlabel('DOS', fontsize=16, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5, zorder=5)
    ax2.legend(loc='upper right', fontsize=14, framealpha=0.8, edgecolor='none', facecolor='white')
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='x', which='major', labelsize=14)
    
    # Set y-axis limits to match
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # Add title (ElectrideFlow style: fontsize=22)
    if title:
        fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)
    
    # Use fixed subplot margins (ElectrideFlow style)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92)
    
    # Save both PNG and PDF
    output_png = output_path
    output_pdf = output_path.with_suffix('.pdf')
    
    fig.savefig(str(output_png), dpi=300)
    print(f"  Saved: {output_png.name}")
    
    fig.savefig(str(output_pdf))
    print(f"  Saved: {output_pdf.name}")
    
    plt.close(fig)
    
    print(f"  Saved: {output_path.name}")

def save_band_structure_data(phonon, structure, output_path):
    """Save phonon band structure data in format compatible with phonon_band_dos_plot.py."""
    kpath = HighSymmKpath(structure)
    kpath_dict = kpath.kpath
    
    # Build q-points for band structure (nested list structure for run_band_structure)
    qpoints_for_phonopy = []
    segment_markers = [0]
    
    for branch in kpath_dict['path']:
        branch_qpoints = []
        for i in range(len(branch) - 1):
            start = np.array(kpath_dict['kpoints'][branch[i]])
            end = np.array(kpath_dict['kpoints'][branch[i+1]])
            
            for j in range(51):
                frac = j / 50.0
                qpoint = start + frac * (end - start)
                branch_qpoints.append(qpoint)
        
        qpoints_for_phonopy.append(branch_qpoints)
        segment_markers.append(segment_markers[-1] + len(branch_qpoints))
    
    # Calculate band structure using run_band_structure (like ElectrideFlow)
    phonon.run_band_structure(qpoints_for_phonopy, is_band_connection=False)
    band_dict = phonon.get_band_structure_dict()
    
    # Flatten results from branches
    all_qpoints = []
    all_frequencies = []
    for qpts, freqs in zip(band_dict['qpoints'], band_dict['frequencies']):
        all_qpoints.extend(qpts)
        all_frequencies.extend(freqs)
    
    all_qpoints = np.array(all_qpoints)
    frequencies_band = np.array(all_frequencies)
    
    # Calculate distances manually in reciprocal space
    rec_lattice = structure.lattice.reciprocal_lattice
    distances_band = [0.0]
    
    for i in range(1, len(all_qpoints)):
        q_prev_cart = rec_lattice.get_cartesian_coords(all_qpoints[i-1])
        q_curr_cart = rec_lattice.get_cartesian_coords(all_qpoints[i])
        delta = np.linalg.norm(q_curr_cart - q_prev_cart)
        distances_band.append(distances_band[-1] + delta)
    
    distances_band = np.array(distances_band)
    
    # Save band data
    with open(output_path, 'w') as f:
        f.write("# Phonon band structure\n")
        f.write("# Column 1: Distance along path\n")
        f.write("# Column 2-4: q-point (fractional coordinates)\n")
        f.write(f"# Columns 5+: Phonon frequencies (THz) for each band\n")
        f.write("#\n")
        f.write(f"# {'Distance':>12}  {'qx':>10} {'qy':>10} {'qz':>10}")
        for i in range(frequencies_band.shape[1]):
            f.write(f"  {'Band'+str(i+1):>12}")
        f.write("\n")
        
        for i in range(len(distances_band)):
            qpt = all_qpoints[i]
            f.write(f"  {distances_band[i]:12.6f}  {qpt[0]:10.6f} {qpt[1]:10.6f} {qpt[2]:10.6f}")
            for band_idx in range(frequencies_band.shape[1]):
                f.write(f"  {frequencies_band[i, band_idx]:12.6f}")
            f.write("\n")
    
    print(f"    Saved: {output_path.name}")
    
    # Save k-path metadata
    kpath_file = output_path.parent / 'band_kpath.dat'
    lattice = structure.lattice.matrix
    
    with open(kpath_file, 'w') as f:
        f.write("# K-path metadata for phonon band structure\n")
        f.write("# Contains lattice, segment structure, and high-symmetry point labels\n")
        f.write("#\n")
        f.write("# LATTICE VECTORS (Angstrom):\n")
        f.write("# Format: LATTICE_A/B/C  x  y  z\n")
        f.write("#\n")
        f.write(f"LATTICE_A  {lattice[0, 0]:12.8f}  {lattice[0, 1]:12.8f}  {lattice[0, 2]:12.8f}\n")
        f.write(f"LATTICE_B  {lattice[1, 0]:12.8f}  {lattice[1, 1]:12.8f}  {lattice[1, 2]:12.8f}\n")
        f.write(f"LATTICE_C  {lattice[2, 0]:12.8f}  {lattice[2, 1]:12.8f}  {lattice[2, 2]:12.8f}\n")
        f.write("#\n")
        f.write("# SEGMENT STRUCTURE:\n")
        f.write("# Format: SEGMENT  seg_idx  start_qidx  end_qidx  n_points\n")
        f.write("#\n")
        
        for seg_idx, (start_idx, end_idx) in enumerate(zip(segment_markers[:-1], segment_markers[1:])):
            n_points = end_idx - start_idx
            f.write(f"SEGMENT  {seg_idx:3d}  {start_idx:6d}  {end_idx-1:6d}  {n_points:6d}\n")
        
        f.write("#\n")
        f.write("# HIGH-SYMMETRY POINTS:\n")
        f.write("# Format: TICK  distance  qx  qy  qz  label\n")
        f.write("#\n")
        
        # Collect tick positions
        tick_positions = [distances_band[0]]
        tick_labels = [convert_label_to_latex(kpath_dict['path'][0][0])]
        
        for branch_idx, branch in enumerate(kpath_dict['path']):
            end_idx = segment_markers[branch_idx + 1] - 1
            if end_idx < len(distances_band):
                tick_positions.append(distances_band[end_idx])
                tick_labels.append(convert_label_to_latex(branch[-1]))
        
        # Write ticks
        for pos, label, qpt in zip(tick_positions, tick_labels, [all_qpoints[0]] + [all_qpoints[segment_markers[i+1]-1] for i in range(len(segment_markers)-1)]):
            f.write(f"TICK     {pos:12.8f}  {qpt[0]:12.8f}  {qpt[1]:12.8f}  {qpt[2]:12.8f}  {label}\n")
    
    print(f"    Saved: {kpath_file.name}")

def save_dos_data(total_dos_dict, proj_dos_dict, primitive, output_path):
    """Save phonon DOS data in format compatible with phonon_band_dos_plot.py."""
    frequencies = total_dos_dict['frequency_points']
    total_dos = total_dos_dict['total_dos']
    
    # Group by element
    element_dos = {}
    for i in range(len(primitive.masses)):
        symbol = primitive.symbols[i]
        pdos = proj_dos_dict['projected_dos'][i]
        
        if symbol in element_dos:
            element_dos[symbol] += pdos
        else:
            element_dos[symbol] = pdos.copy()
    
    with open(output_path, 'w') as f:
        f.write("# Phonon Density of States (DOS)\n")
        f.write("# Column 1: Frequency (THz)\n")
        f.write("# Column 2: Total DOS\n")
        for idx, element in enumerate(element_dos.keys()):
            f.write(f"# Column {idx+3}: {element} projected DOS\n")
        f.write("#\n")
        f.write(f"{'Frequency':>12}  {'Total':>12}")
        for element in element_dos.keys():
            f.write(f"  {element:>12}")
        f.write("\n")
        
        for i, freq in enumerate(frequencies):
            f.write(f"  {freq:12.6f}  {total_dos[i]:12.6f}")
            for pdos in element_dos.values():
                f.write(f"  {pdos[i]:12.6f}")
            f.write("\n")
    
    print(f"    Saved: {output_path.name}")

def calculate_thermal_properties(phonon, structure, output_path):
    """Calculate and save thermal properties using phonopy."""
    from phonopy.phonon.thermal_properties import ThermalProperties
    
    # Get thermal properties
    tp = ThermalProperties(phonon.get_frequencies(), 
                           cutoff_frequency=0.0,
                           pretend_real=True)
    
    # Temperature range: 0-1000 K, step 10 K
    temperatures = np.arange(0, 1001, 10)
    tp.run(t_step=10, t_max=1000, t_min=0)
    
    temps, free_energy, entropy, heat_capacity = tp.get_thermal_properties()
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("# Thermal properties from phonon DOS\n")
        f.write("# T: Temperature (K)\n")
        f.write("# F: Helmholtz free energy (kJ/mol)\n")
        f.write("# S: Entropy (J/K/mol)\n")
        f.write("# Cv: Heat capacity at constant volume (J/K/mol)\n")
        f.write("# U: Internal energy (kJ/mol)\n")
        f.write("#\n")
        f.write(f"{'T (K)':>10} {'F (kJ/mol)':>15} {'S (J/K/mol)':>15} {'Cv (J/K/mol)':>15} {'U (kJ/mol)':>15}\n")
        
        for i, T in enumerate(temps):
            # Convert units: phonopy outputs in eV and eV/K
            F_kJ = free_energy[i] * 96.485  # eV to kJ/mol
            S_J = entropy[i] * 96.485  # eV/K to J/(K·mol)
            Cv_J = heat_capacity[i] * 96.485  # eV/K to J/(K·mol)
            U_kJ = (free_energy[i] + T * entropy[i] / 1000.0) * 96.485  # U = F + TS
            
            f.write(f"{T:10.1f} {F_kJ:15.6f} {S_J:15.6f} {Cv_J:15.6f} {U_kJ:15.6f}\n")
    
    print(f"    Saved: {output_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze displaced structures and post-process phonon calculations"
    )
    parser.add_argument(
        '--collect-forces',
        action='store_true',
        help="Explicitly collect forces (auto-detected if in phonon directory)"
    )
    parser.add_argument(
        '--mesh',
        type=int,
        nargs=3,
        default=[20, 20, 20],
        help="Mesh for DOS calculation (default: 20 20 20)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in a phonon calculation directory
    current_dir = Path('.')
    is_phonon_dir = (
        (current_dir / 'phonopy_disp.yaml').exists() or 
        (current_dir / 'phonopy.yaml').exists()
    ) and list(current_dir.glob('[0-9][0-9][0-9]'))
    
    if is_phonon_dir or args.collect_forces:
        # Mode: Direct phonon processing in calculation directory
        print("="*70)
        print("PHONON CALCULATION POST-PROCESSING")
        print("="*70)
        print(f"Working directory: {current_dir.absolute()}")
        
        # Collect forces and generate force constants
        if args.collect_forces or not (current_dir / 'phonopy_params.yaml').exists():
            print("\nCollecting forces from displacement calculations...")
            
            # Find all displacement directories
            disp_dirs = sorted(current_dir.glob('[0-9][0-9][0-9]'))
            print(f"  Found {len(disp_dirs)} displacement directories")
            
            # Check for vasprun.xml files
            vasprun_paths = []
            for disp_dir in disp_dirs:
                vasprun = disp_dir / 'vasprun.xml'
                if vasprun.exists():
                    vasprun_paths.append(vasprun)
            
            print(f"  Found {len(vasprun_paths)} vasprun.xml files")
            
            if len(vasprun_paths) < len(disp_dirs):
                print(f"  WARNING: Missing {len(disp_dirs) - len(vasprun_paths)} vasprun.xml files")
            
            if not vasprun_paths:
                print("\nERROR: No vasprun.xml files found")
                return
            
            # Load structure
            poscar_path = current_dir / 'POSCAR'
            if not poscar_path.exists():
                print("\nERROR: POSCAR not found")
                return
            
            structure = Structure.from_file(str(poscar_path))
            composition = structure.composition.reduced_formula
            print(f"  Structure: {composition} ({len(structure)} atoms)")
            
            # Extract forces using Python API
            print("\n  Extracting forces from vasprun.xml files...")
            try:
                forces = extract_forces_from_vasprun(vasprun_paths)
                print(f"    Extracted forces: shape {forces.shape}")
            except Exception as e:
                print(f"\nERROR: Force extraction failed: {e}")
                return
            
            # Determine supercell matrix from phonopy_disp.yaml
            disp_yaml = current_dir / 'phonopy_disp.yaml'
            if not disp_yaml.exists():
                disp_yaml = current_dir / 'phonopy.yaml'
            
            if disp_yaml.exists():
                # Load to get supercell matrix
                temp_phonon = load(str(disp_yaml))
                supercell_matrix = temp_phonon.supercell_matrix
                print(f"    Supercell matrix: {np.diag(supercell_matrix).astype(int)}")
            else:
                # Default supercell
                supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
                print(f"    Using default supercell matrix: 2 2 2")
            
            # Create force constants using Python API
            print("\n  Creating force constants...")
            try:
                phonon = create_force_constants(structure, current_dir, forces, supercell_matrix)
                print("    Created: phonopy_params.yaml")
            except Exception as e:
                print(f"\nERROR: Force constants generation failed: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            # Load existing phonopy data
            print("\nLoading existing phonopy data...")
            phonon = load_phonopy_data(current_dir)
            
            # Load structure
            poscar_path = current_dir / 'POSCAR'
            structure = Structure.from_file(str(poscar_path))
            composition = structure.composition.reduced_formula
            print(f"  Structure: {composition} ({len(structure)} atoms)")
        
        # Check Gamma point frequencies
        print("\nChecking Gamma point frequencies...")
        gamma_ok = check_gamma_frequencies(phonon)
        
        # Generate data files
        print("\nGenerating data files...")
        band_data_file = current_dir / "phonon_band.dat"
        save_band_structure_data(phonon, structure, band_data_file)
        
        # Calculate and save DOS data
        mesh = args.mesh
        print(f"\nCalculating DOS (mesh: {mesh})...")
        phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
        phonon.run_total_dos()
        phonon.run_projected_dos()
        
        total_dos_dict = phonon.get_total_dos_dict()
        proj_dos_dict = phonon.get_projected_dos_dict()
        primitive = phonon.primitive
        
        dos_data_file = current_dir / "phonon_dos.dat"
        save_dos_data(total_dos_dict, proj_dos_dict, primitive, dos_data_file)
        
        # Calculate thermal properties
        print("\nCalculating thermal properties...")
        thermal_file = current_dir / "thermal.dat"
        calculate_thermal_properties(phonon, structure, thermal_file)
        
        # Generate plots
        print("\nGenerating band structure plot...")
        band_output = current_dir / "phonon_band.png"
        plot_band_structure(phonon, structure, band_output, 
                          title=f"{composition} Phonon Band Structure")
        
        print("\nGenerating DOS plot...")
        dos_output = current_dir / "phonon_dos.png"
        plot_dos(phonon, structure, dos_output, mesh=mesh)
        
        print("\nGenerating combined band+DOS plot...")
        combined_output = current_dir / "phonon_band_dos.png"
        plot_combined_band_dos(phonon, structure, combined_output,
                             title=f"{composition} Phonon", mesh=mesh)
        
        print("\n" + "="*70)
        print("PHONON POST-PROCESSING COMPLETE")
        print("="*70)
        
        if gamma_ok:
            print("All soft modes eliminated!")
        else:
            print("Soft modes still present - may need further relaxation")
        
        print(f"\nGenerated files:")
        print(f"  - phonopy_params.yaml     (force constants)")
        print(f"  - phonon_band.dat         (band structure data)")
        print(f"  - band_kpath.dat          (k-path metadata)")
        print(f"  - phonon_dos.dat          (DOS data)")
        print(f"  - thermal.dat             (thermal properties)")
        print(f"  - phonon_band.png         (band plot)")
        print(f"  - phonon_dos.png          (DOS plot)")
        print(f"  - phonon_band_dos.png     (combined plot)")
        print("="*70)
        return
    
    # If not in phonon directory, show error
    print("\n" + "="*70)
    print("ERROR: Not in a phonon calculation directory")
    print("="*70)
    print("\nThis script must be run in a directory containing:")
    print("  - phonopy_disp.yaml or phonopy.yaml")
    print("  - Displacement directories: 001/, 002/, 003/, etc.")
    print("\nUsage:")
    print("  cd /path/to/PHON_mode1")
    print("  /path/to/analyze_phon.py")
    print("\nOr with explicit flag:")
    print("  /path/to/analyze_phon.py --collect-forces")
    print("="*70)

if __name__ == "__main__":
    main()
