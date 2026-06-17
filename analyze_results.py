
import csv
import os
from collections import defaultdict
import statistics

file1_path = r"c:\Users\Cathy\Documents\scripts\wafermap_anomaly_detection_result\patchcore-inspection-base_working20251124\results\repeat\anomaly_classifications_b3_image256_6464_1vdensity.csv"
file2_path = r"c:\Users\Cathy\Documents\scripts\wafermap_anomaly_detection_result\patchcore-inspection-base_working20251124\results\repeat\anomaly_classifications_b2_image256_6464_3vdensity.csv"

def analyze_csv(file_path):
    counts = defaultdict(int)
    random_probs = []
    predictions = {} # image -> (class, prob)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return counts, random_probs, predictions

    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['image']
            cls = row['pred_class']
            prob = float(row['prob'])
            
            counts[cls] += 1
            predictions[img] = (cls, prob)
            
            if cls == 'Random':
                random_probs.append(prob)
                
    return counts, random_probs, predictions

print(f"Analyzing File 1 (1-vector): {os.path.basename(file1_path)}")
counts1, probs1, preds1 = analyze_csv(file1_path)
print("Class Counts:", dict(counts1))
if probs1:
    print(f"Random Avg Prob: {statistics.mean(probs1):.4f} (n={len(probs1)})")
else:
    print("No Random predictions.")

print(f"\nAnalyzing File 2 (3-vector): {os.path.basename(file2_path)}")
counts2, probs2, preds2 = analyze_csv(file2_path)
print("Class Counts:", dict(counts2))
if probs2:
    print(f"Random Avg Prob: {statistics.mean(probs2):.4f} (n={len(probs2)})")
else:
    print("No Random predictions.")

# Compare changes
print("\n--- Comparison (1-vector -> 3-vector) ---")
common_images = set(preds1.keys()) & set(preds2.keys())
print(f"Common images: {len(common_images)}")

random_gained = 0
random_lost = 0
nearfull_to_random = 0
random_to_nearfull = 0

for img in common_images:
    cls1, prob1 = preds1[img]
    cls2, prob2 = preds2[img]
    
    if cls1 != cls2:
        if cls2 == 'Random':
            random_gained += 1
            if cls1 == 'Near-full':
                nearfull_to_random += 1
        if cls1 == 'Random':
            random_lost += 1
            if cls2 == 'Near-full':
                random_to_nearfull += 1

print(f"Random predictions gained: {random_gained}")
print(f"Random predictions lost: {random_lost}")
print(f"Near-full -> Random switches: {nearfull_to_random}")
print(f"Random -> Near-full switches: {random_to_nearfull}")

