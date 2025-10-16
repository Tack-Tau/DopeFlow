#!/usr/bin/env python3
"""
Electride Analysis from VASP ELFCAR files

This script analyzes the Electron Localization Function (ELF) from VASP calculations
to identify potential electride structures using Bader topological analysis.
Electrides are materials where electrons occupy interstitial regions rather than 
being associated with atoms.

Uses Bader topological analysis to identify critical points in the ELF field.
Requires the Bader program from: https://theory.cm.utexas.edu/henkelman/code/bader/

Reference: https://github.com/MaterSim/CMS/blob/master/Scripts/Pymatgen/ELF.py

Usage:
    python3 analyze_electride.py <path_to_ELFCAR>
    python3 analyze_electride.py --batch <directory_with_ELF_subdirs>
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path

try:
    from pymatgen.io.vasp import Chgcar
    from pymatgen.core import Structure
except ImportError:
    print("Error: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)

try:
    import scipy
except ImportError:
    print("Error: scipy is required. Install with: pip install scipy")
    sys.exit(1)


class ElectrideAnalyzer:
    """
    Analyzer for detecting electride characteristics from ELFCAR files.
    """
    
    def __init__(
        self, 
        elf_threshold=0.6,
        min_distance_from_atoms=1.5,
        volume_threshold=0.5,
        bader_executable='bader',
        verbose=True
    ):
        """
        Initialize the electride analyzer using Bader topological analysis.
        
        Parameters:
        -----------
        elf_threshold : float
            Minimum ELF value to consider as potential electride site (default: 0.6)
            Typical values: 0.5-0.7
        min_distance_from_atoms : float
            Minimum distance (in Angstroms) from atomic positions to consider
            as interstitial region (default: 1.5)
        volume_threshold : float
            Minimum volume fraction (in Angstrom^3) for an interstitial region
            to be significant (default: 0.5)
        bader_executable : str
            Path to bader executable (default: 'bader')
        verbose : bool
            Print detailed analysis information
        """
        self.elf_threshold = elf_threshold
        self.min_distance_from_atoms = min_distance_from_atoms
        self.volume_threshold = volume_threshold
        self.bader_executable = bader_executable
        self.verbose = verbose
        
    def read_elfcar(self, elfcar_path):
        """
        Read ELFCAR file using pymatgen.
        
        Parameters:
        -----------
        elfcar_path : str or Path
            Path to ELFCAR file
            
        Returns:
        --------
        structure : Structure
            Pymatgen Structure object
        elf_data : np.ndarray
            3D array of ELF values
        """
        elfcar_path = Path(elfcar_path)
        if not elfcar_path.exists():
            raise FileNotFoundError(f"ELFCAR file not found: {elfcar_path}")
            
        if self.verbose:
            print(f"Reading ELFCAR from: {elfcar_path}")
            
        # Read ELFCAR using Chgcar class (ELF has same format)
        elfcar = Chgcar.from_file(str(elfcar_path))
        structure = elfcar.structure
        elf_data = elfcar.data['total']
        
        if self.verbose:
            print(f"Structure: {structure.composition.reduced_formula}")
            print(f"Grid dimensions: {elf_data.shape}")
            print(f"ELF range: [{elf_data.min():.4f}, {elf_data.max():.4f}]")
            
        return structure, elf_data
    
    def estimate_interstitial_volume(self, elf_data, structure):
        """
        Estimate the volume of high-ELF interstitial regions.
        
        Parameters:
        -----------
        elf_data : np.ndarray
            3D array of ELF values
        structure : Structure
            Pymatgen Structure object
            
        Returns:
        --------
        volume : float
            Volume of interstitial regions in Angstrom^3
        volume_fraction : float
            Fraction of unit cell volume occupied by interstitial electrons
        """
        # Count voxels above threshold
        high_elf_voxels = np.sum(elf_data > self.elf_threshold)
        total_voxels = np.prod(elf_data.shape)
        
        # Calculate volume
        unit_cell_volume = structure.lattice.volume
        voxel_volume = unit_cell_volume / total_voxels
        interstitial_volume = high_elf_voxels * voxel_volume
        volume_fraction = high_elf_voxels / total_voxels
        
        return interstitial_volume, volume_fraction
    
    def run_bader_analysis(self, elfcar_path):
        """
        Run Bader topological analysis on ELFCAR file.
        
        Parameters:
        -----------
        elfcar_path : Path
            Path to ELFCAR file
            
        Returns:
        --------
        bcf_path : Path
            Path to generated BCF.dat file, or None if failed
        """
        elfcar_path = Path(elfcar_path)
        work_dir = elfcar_path.parent
        
        if self.verbose:
            print(f"Running Bader analysis on {elfcar_path.name}...")
        
        # Determine bader executable path
        bader_exe = self.bader_executable
        if bader_exe == 'bader' or not Path(bader_exe).is_absolute():
            local_bader = work_dir / 'bader'
            if local_bader.exists() and local_bader.is_file():
                bader_exe = str(local_bader)
                if self.verbose:
                    print(f"Found bader executable in ELFCAR directory: {local_bader.name}")
        
        try:
            result = subprocess.run(
                [bader_exe, '-h'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"Bader executable '{bader_exe}' not found or not working.\n"
                f"Solutions:\n"
                f"  1. Place 'bader' executable in same directory as ELFCAR\n"
                f"  2. Add bader to system PATH\n"
                f"  3. Use --bader-exe to specify path\n"
                f"Download from: https://theory.cm.utexas.edu/henkelman/code/bader/"
            )
        
        # Run bader analysis
        try:
            result = subprocess.run(
                [bader_exe, str(elfcar_path.name)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Bader analysis failed: {result.stderr}")
            
            bcf_path = work_dir / "BCF.dat"
            if not bcf_path.exists():
                raise RuntimeError("BCF.dat not generated by Bader analysis")
            
            if self.verbose:
                print(f"Bader analysis complete. BCF.dat generated.")
            
            return bcf_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Bader analysis timed out (>5 minutes)")
    
    def parse_bcf_file(self, bcf_path):
        """
        Parse BCF.dat file from Bader analysis.
        
        Parameters:
        -----------
        bcf_path : Path
            Path to BCF.dat file
            
        Returns:
        --------
        maxima_coords : np.ndarray
            Cartesian coordinates of Bader maxima (Nx3)
        maxima_elf : np.ndarray
            ELF values at maxima (N,)
        distances_from_atoms : np.ndarray
            Distance to nearest atom for each maximum (N,)
        """
        bcf_path = Path(bcf_path)
        
        if not bcf_path.exists():
            raise FileNotFoundError(f"BCF.dat file not found: {bcf_path}")
        
        # Read BCF.dat
        # Format: # X Y Z CHARGE ATOM DISTANCE
        coords = []
        elf_values = []
        distances = []
        
        with open(bcf_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header and separator lines
                if line.startswith('#') or line.startswith('-') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        # Parse: index, x, y, z, charge(ELF), atom_index, distance
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        elf = float(parts[4])
                        dist = float(parts[6])
                        
                        coords.append([x, y, z])
                        elf_values.append(elf)
                        distances.append(dist)
                    except (ValueError, IndexError):
                        continue
        
        if len(coords) == 0:
            raise ValueError("No valid data found in BCF.dat")
        
        maxima_coords = np.array(coords)
        maxima_elf = np.array(elf_values)
        distances_from_atoms = np.array(distances)
        
        if self.verbose:
            print(f"Parsed {len(maxima_coords)} Bader maxima from BCF.dat")
        
        return maxima_coords, maxima_elf, distances_from_atoms
    
    def filter_bader_interstitial_maxima(self, maxima_coords, maxima_elf, distances):
        """
        Filter Bader maxima to identify interstitial sites.
        
        Parameters:
        -----------
        maxima_coords : np.ndarray
            Coordinates of all Bader maxima
        maxima_elf : np.ndarray
            ELF values at maxima
        distances : np.ndarray
            Distances to nearest atoms
            
        Returns:
        --------
        interstitial_coords : np.ndarray
            Coordinates of interstitial maxima
        interstitial_elf : np.ndarray
            ELF values at interstitial maxima
        interstitial_distances : np.ndarray
            Distances from atoms for interstitial maxima
        """
        # Filter based on distance from atoms and ELF threshold
        interstitial_mask = (
            (distances > self.min_distance_from_atoms) &
            (maxima_elf > self.elf_threshold)
        )
        
        interstitial_coords = maxima_coords[interstitial_mask]
        interstitial_elf = maxima_elf[interstitial_mask]
        interstitial_distances = distances[interstitial_mask]
        
        if self.verbose:
            print(f"Found {len(interstitial_coords)} interstitial Bader maxima")
            print(f"  (distance > {self.min_distance_from_atoms} Å, ELF > {self.elf_threshold})")
            if len(interstitial_elf) > 0:
                print(f"  Max interstitial ELF: {np.max(interstitial_elf):.4f}")
                print(f"  Mean interstitial distance: {np.mean(interstitial_distances):.4f} Å")
        
        return interstitial_coords, interstitial_elf, interstitial_distances
    
    def analyze(self, elfcar_path):
        """
        Perform complete electride analysis on an ELFCAR file.
        
        Parameters:
        -----------
        elfcar_path : str or Path
            Path to ELFCAR file
            
        Returns:
        --------
        results : dict
            Dictionary containing analysis results
        """
        elfcar_path = Path(elfcar_path)
        
        # Read ELFCAR for structure info and volume calculation
        structure, elf_data = self.read_elfcar(elfcar_path)
        
        # Bader topological analysis
        if self.verbose:
            print("\n=== Bader Topological Analysis ===")
        
        # Check if BCF.dat already exists
        bcf_path = elfcar_path.parent / "BCF.dat"
        if bcf_path.exists():
            if self.verbose:
                print(f"Using existing BCF.dat file (delete to regenerate)")
        else:
            # Run Bader analysis
            bcf_path = self.run_bader_analysis(elfcar_path)
        
        # Parse BCF.dat
        maxima_coords, maxima_elf, distances = self.parse_bcf_file(bcf_path)
        
        # Filter interstitial maxima
        interstitial_coords, interstitial_values, interstitial_distances = \
            self.filter_bader_interstitial_maxima(maxima_coords, maxima_elf, distances)
        
        # Estimate interstitial volume
        volume, volume_fraction = self.estimate_interstitial_volume(elf_data, structure)
        
        # Determine if structure is a potential electride
        is_electride = (
            len(interstitial_coords) > 0 and
            np.max(interstitial_values) > self.elf_threshold and
            volume > self.volume_threshold
        )
        
        # Compile results
        results = {
            'is_potential_electride': is_electride,
            'structure_formula': structure.composition.reduced_formula,
            'max_elf': float(elf_data.max()),
            'max_interstitial_elf': float(np.max(interstitial_values)) if len(interstitial_values) > 0 else 0.0,
            'n_interstitial_sites': len(interstitial_coords),
            'interstitial_volume': float(volume),
            'interstitial_volume_fraction': float(volume_fraction),
            'unit_cell_volume': float(structure.lattice.volume),
            'interstitial_elf_values': interstitial_values.tolist() if len(interstitial_values) > 0 else [],
            'interstitial_distances_from_atoms': interstitial_distances.tolist() if len(interstitial_distances) > 0 else []
        }
        
        # Print summary
        if self.verbose:
            self._print_results(results)
            
        return results
    
    def _print_results(self, results):
        """Print formatted analysis results."""
        print("\n" + "="*70)
        print("ELECTRIDE ANALYSIS RESULTS (Bader Topological)")
        print("="*70)
        print(f"Formula: {results['structure_formula']}")
        print(f"Maximum ELF value: {results['max_elf']:.4f}")
        print(f"Maximum interstitial ELF: {results['max_interstitial_elf']:.4f}")
        print(f"Number of interstitial sites: {results['n_interstitial_sites']}")
        print(f"Interstitial volume: {results['interstitial_volume']:.4f} Å³")
        print(f"Volume fraction: {results['interstitial_volume_fraction']:.4f}")
        print(f"Unit cell volume: {results['unit_cell_volume']:.4f} Å³")
        print("-"*70)
        
        if results['is_potential_electride']:
            print("  POTENTIAL ELECTRIDE DETECTED")
            print(f"  High ELF values ({results['max_interstitial_elf']:.3f}) found in")
            print(f"  {results['n_interstitial_sites']} interstitial region(s)")
        else:
            print("  NOT A LIKELY ELECTRIDE")
            if results['n_interstitial_sites'] == 0:
                print("  No significant interstitial ELF maxima found")
            elif results['max_interstitial_elf'] < self.elf_threshold:
                print(f"  Interstitial ELF too low ({results['max_interstitial_elf']:.3f})")
            else:
                print(f"  Insufficient interstitial volume ({results['interstitial_volume']:.3f} Å³)")
        print("="*70 + "\n")
    
    def batch_analyze(self, parent_directory, output_file='electride_analysis.csv'):
        """
        Analyze multiple ELFCAR files in subdirectories.
        
        Parameters:
        -----------
        parent_directory : str or Path
            Parent directory containing subdirectories with ELF calculations
        output_file : str
            Output CSV file for batch results
            
        Returns:
        --------
        all_results : list
            List of dictionaries with results for each structure
        """
        parent_dir = Path(parent_directory)
        all_results = []
        
        # Find all ELFCAR files
        elfcar_files = list(parent_dir.rglob("*/ELF/ELFCAR"))
        
        if len(elfcar_files) == 0:
            print(f"No ELFCAR files found in {parent_dir}")
            return []
        
        print(f"Found {len(elfcar_files)} ELFCAR files to analyze\n")
        
        for i, elfcar_path in enumerate(elfcar_files, 1):
            print(f"\nAnalyzing {i}/{len(elfcar_files)}: {elfcar_path.parent}")
            print("-"*70)
            
            try:
                results = self.analyze(elfcar_path)
                results['path'] = str(elfcar_path.parent)
                results['structure_id'] = elfcar_path.parent.parent.name
                all_results.append(results)
            except Exception as e:
                print(f"Error analyzing {elfcar_path}: {e}")
                all_results.append({
                    'path': str(elfcar_path.parent),
                    'structure_id': elfcar_path.parent.parent.name,
                    'is_potential_electride': False,
                    'error': str(e)
                })
        
        # Save results to CSV
        self._save_to_csv(all_results, output_file)
        
        # Print summary
        electrides = [r for r in all_results if r.get('is_potential_electride', False)]
        print(f"\n{'='*70}")
        print(f"BATCH ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"Total structures analyzed: {len(all_results)}")
        print(f"Potential electrides found: {len(electrides)}")
        print(f"Results saved to: {output_file}")
        
        if electrides:
            print(f"\nPotential electride structures:")
            for r in electrides:
                print(f"  - {r['structure_id']}: {r['structure_formula']} "
                      f"(ELF={r['max_interstitial_elf']:.3f})")
        print(f"{'='*70}\n")
        
        return all_results
    
    def _save_to_csv(self, results, output_file):
        """Save results to CSV file."""
        import csv
        
        if not results:
            return
        
        # Define fields to save
        fields = [
            'structure_id', 'path', 'structure_formula', 'is_potential_electride',
            'max_elf', 'max_interstitial_elf', 'n_interstitial_sites',
            'interstitial_volume', 'interstitial_volume_fraction', 'unit_cell_volume'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze VASP ELFCAR files to identify potential electride structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single structure analysis
  python3 analyze_electride.py /path/to/structure/ELF/ELFCAR
  
  # Batch analysis of all structures
  python3 analyze_electride.py --batch /path/to/parent/directory
  
  # Adjust detection parameters
  python3 analyze_electride.py ELFCAR --threshold 0.7 --min-distance 2.0
  
  # Use custom bader executable path
  python3 analyze_electride.py --bader-exe /path/to/bader /path/to/structure/ELFCAR
  
  # Force regenerate BCF.dat (delete it first)
  rm /path/to/ELF/BCF.dat
  python3 analyze_electride.py /path/to/structure/ELFCAR
        """
    )
    
    parser.add_argument(
        'elfcar_path',
        nargs='?',
        help='Path to ELFCAR file or parent directory (with --batch)'
    )
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Batch mode: analyze all ELFCAR files in subdirectories'
    )
    parser.add_argument(
        '--bader-exe',
        default='bader',
        help='Path to bader executable (default: bader)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.6,
        help='ELF threshold for electride detection (default: 0.6)'
    )
    parser.add_argument(
        '--min-distance', '-d',
        type=float,
        default=1.5,
        help='Minimum distance from atoms in Angstroms (default: 1.5)'
    )
    parser.add_argument(
        '--volume-threshold', '-v',
        type=float,
        default=0.5,
        help='Minimum interstitial volume in Angstrom^3 (default: 0.5)'
    )
    parser.add_argument(
        '--output', '-o',
        default='electride_analysis.csv',
        help='Output CSV file for batch analysis (default: electride_analysis.csv)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    if args.elfcar_path is None:
        # Default to current directory if no path provided
        args.elfcar_path = '.'
    
    # Initialize analyzer
    analyzer = ElectrideAnalyzer(
        elf_threshold=args.threshold,
        min_distance_from_atoms=args.min_distance,
        volume_threshold=args.volume_threshold,
        bader_executable=args.bader_exe,
        verbose=not args.quiet
    )
    
    # Perform analysis
    if args.batch:
        analyzer.batch_analyze(args.elfcar_path, args.output)
    else:
        # Check if path is a file or directory containing ELFCAR
        elfcar_path = Path(args.elfcar_path)
        if elfcar_path.is_dir():
            elfcar_path = elfcar_path / 'ELFCAR'
        
        results = analyzer.analyze(elfcar_path)
        
        # Return appropriate exit code
        sys.exit(0 if results['is_potential_electride'] else 1)


if __name__ == '__main__':
    main()

