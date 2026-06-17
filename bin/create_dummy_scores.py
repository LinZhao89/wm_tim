import os
import csv
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Create a dummy image scores CSV with 0 scores.")
    parser.add_argument("data_root", help="Root folder to scan (e.g. dataset/wm811k/mxwm38/test)")
    parser.add_argument("output_csv", help="Output CSV path")
    args = parser.parse_args()

    rows = []
    
    # Walk through the directory
    print(f"Scanning {args.data_root}...")
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            if file.lower().endswith(".png"):
                # Get the full path relative to the current working directory (if data_root is relative)
                full_path = os.path.join(root, file)
                
                # Normalize to forward slashes to match the requested format
                # and ensure it looks like a relative path from the project root
                # (assuming the script is run from project root and data_root is passed as relative)
                formatted_path = full_path.replace("\\", "/")
                
                rows.append([formatted_path, 1])
    
    if not rows:
        print("No PNG images found! Check the path.")
        return

    # Sort rows by path for consistency
    rows.sort(key=lambda x: x[0])

    # Write to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "image_score_mean"])
        writer.writerows(rows)
        
    print(f"Successfully created {args.output_csv} with {len(rows)} entries.")

if __name__ == "__main__":
    main()
