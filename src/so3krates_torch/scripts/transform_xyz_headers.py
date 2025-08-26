import re

# Mapping of replacements
replacements = {
    "hirsh_ratios": "REF_hirsh_ratios",
    "forces": "REF_forces",
    "energy": "REF_energy",
    "dipole_total": "REF_dipole_total",
    "dipole": "REF_dipole",
}


def process_xyz(input_file, output_file):
    with open(input_file, "r") as f:
        text = f.read()

    # Replace only whole words to avoid partial matches
    for old, new in replacements.items():
        text = re.sub(rf"\b{old}\b", new, text)

    with open(output_file, "w") as f:
        f.write(text)


process_xyz(
    "/home/thenkes/Documents/Uni/Promotion/Research/torchkrates/So3krates-torch/development_junk/md17_ethanol_small.xyz",
    "/home/thenkes/Documents/Uni/Promotion/Research/torchkrates/So3krates-torch/development_junk/training/md17_ethanol_small.xyz",
)
