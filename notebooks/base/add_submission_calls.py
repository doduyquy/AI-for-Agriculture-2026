import json
import os

nb_path = r"d:\HocTap\NCKH_ThayDoNhuTai\Challenges\Notebooks\00_baseline\BaselineHS_TopK.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last cell which should be the one defining generate_submission
found = False
for cell in nb['cells'][::-1]: # Search backwards
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = "".join(source)
        if "def generate_submission" in source_str:
            print("Found target cell defining generate_submission.")
            
            # Check if calls already exist to avoid duplication
            if "generate_submission(k, 'scratch', top_idx)" in source_str:
                print("Submission calls appear to already be present. Skipping.")
                found = True # Treat as found/handled
                break

            # Append the calls
            new_code = [
                "\n",
                "# --- EXECUTE SUBMISSION GENERATION ---\n",
                "print(\"\\n=== Generating Submissions ===\")\n",
                "for k in [10, 20, 30]:\n",
                "    if 'sorted_indices' not in globals():\n",
                "        print(\"Error: sorted_indices not found. Make sure previous cells were run.\")\n",
                "        break\n",
                "    top_idx = sorted(sorted_indices[:k])\n",
                "    \n",
                "    # Generate for Scratch\n",
                "    generate_submission(k, 'scratch', top_idx)\n",
                "    \n",
                "    # Generate for Finetune\n",
                "    generate_submission(k, 'finetune', top_idx)\n"
            ]
            
            cell['source'].extend(new_code)
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully modified {nb_path}")
else:
    print("Could not find the cell defining generate_submission!")
