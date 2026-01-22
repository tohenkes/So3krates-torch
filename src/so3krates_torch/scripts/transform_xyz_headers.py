import re
import os

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


folder = "/home/thenkes/Downloads/testing_data/"
file_list = [
    filename for filename in os.listdir(folder) if filename.endswith("xyz")
]

for input_file in file_list:
    print(f"Processing {input_file}")
    output_file = (
        f"/home/thenkes/Downloads/testing_data/processed_{input_file}"
    )
    process_xyz(f"{folder}/{input_file}", output_file)
