import argparse
from pathlib import Path

# Template for the Phenix .eff file with placeholders for MTZ paths
EFF_TEMPLATE = """
data_manager {{
  miller_array {{
    file = "{sf_mtz}"
    labels {{
      name = "F(+),SIGF(+),F(-),SIGF(-)"
      array_type = unknown *amplitude bool complex hendrickson_lattman \\
                   integer intensity nonsense
    }}
    user_selected_labels = "F(+),SIGF(+),F(-),SIGF(-)"
  }}
  miller_array {{
    file = "{rfree_mtz}"
    labels {{
      name = "R-free-flags"
      array_type = unknown amplitude bool complex hendrickson_lattman \\
                   *integer intensity nonsense
    }}
    user_selected_labels = "R-free-flags"
  }}
  fmodel {{
    xray_data {{
      outliers_rejection = False
      french_wilson_scale = False
    }}
  }}

  model {{
    file = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/9b7c.pdb"
  }}
  default_model = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/9b7c.pdb"
}}

refinement {{
  crystal_symmetry {{
    unit_cell   = 79.424  79.424  37.793  90.00  90.00  90.00
    space_group = P 43 21 2
  }}
  refine {{
    strategy = *individual_sites individual_sites_real_space *rigid_body \\
               *individual_adp group_adp tls occupancies group_anomalous den
    adp {{
      individual {{
        isotropic   = None
        anisotropic = None
      }}
      group_adp_refinement_mode = *one_adp_group_per_residue \\
                                    two_adp_groups_per_residue group_selection
      group = None
      tls   = None
    }}
  }}
  main {{
    number_of_macro_cycles = 3
  }}
}}

output {{
  prefix    = "{phenix_out_mtz}"
  serial    = 1
  overwrite = True
}}
"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Phenix .eff file with dynamic MTZ paths."
    )
    parser.add_argument(
        "--sf-mtz", required=True,
        help="Path to the structure-factor MTZ file"
    )
    parser.add_argument(
        "--rfree-mtz", required=True,
        help="Path to the R-free flags MTZ file"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for the generated .eff file"
    )
    parser.add_argument(
        "--phenix-out-mtz", required=True,
        help="Specific name of the output .mtz file of phenix."
    )
    args = parser.parse_args()

    # Fill in the template and write to the output file
    content = EFF_TEMPLATE.format(
        sf_mtz=args.sf_mtz,
        rfree_mtz=args.rfree_mtz,
        phenix_out_mtz=args.phenix_out_mtz
    )
    Path(args.out).write_text(content)
    print(f"Wrote .eff file to {args.out}")

if __name__ == "__main__":
    main()
