"""
analyze_errors.py - PHÂN TÍCH CHI TIẾT CÁC DỰ ĐOÁN SAI

🎯 Mục đích:
  - Phân tích predictions.csv để tìm pattern lỗi
  - Identify các loại lỗi phổ biến nhất
  - Đưa ra khuyến nghị cải thiện model

📊 Output:
  - Confusion matrix
  - Error breakdown by class
  - Top error patterns
  - List of misclassified images
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# Load predictions
predictions_file = Path('predictions.csv')

if not predictions_file.exists():
    print("❌ File predictions.csv không tồn tại!")
    print("   Hãy chạy PipelineV1_Stage1_TwoStep_Inference.py trước")
    exit(1)

df = pd.read_csv(predictions_file)

print("\n" + "="*70)
print("📊 PHÂN TÍCH CHI TIẾT CÁC DỰ ĐOÁN SAI")
print("="*70)

# ==================== TỔNG QUAN ====================
total = len(df)
correct_count = (df['Correct'] == 'Yes').sum()
incorrect_count = (df['Correct'] == 'No').sum()

print(f"\n📈 TỔNG QUAN:")
print(f"  Total samples: {total}")
print(f"  ✅ Correct: {correct_count} ({correct_count/total*100:.1f}%)")
print(f"  ❌ Incorrect: {incorrect_count} ({incorrect_count/total*100:.1f}%)")
print(f"  🎯 Accuracy: {correct_count/total:.3f}")

# ==================== PHÂN TÍCH THEO TRUE CLASS ====================
print(f"\n" + "="*70)
print("❌ PHÂN TÍCH LỖI THEO TRUE CLASS")
print("="*70)

for true_class in ['Health', 'Rust', 'Other']:
    class_df = df[df['True_Class'] == true_class]
    class_correct = class_df[class_df['Correct'] == 'Yes']
    class_wrong = class_df[class_df['Correct'] == 'No']
    
    print(f"\n🔹 {true_class.upper()} (Total: {len(class_df)})")
    print(f"  ✅ Correct: {len(class_correct):3d} ({len(class_correct)/len(class_df)*100:5.1f}%)")
    print(f"  ❌ Wrong:   {len(class_wrong):3d} ({len(class_wrong)/len(class_df)*100:5.1f}%)")
    
    if len(class_wrong) > 0:
        # Nhầm thành class nào?
        wrong_as = class_wrong['Predicted_Class'].value_counts()
        print(f"  📊 Nhầm thành:")
        for pred_class, count in wrong_as.items():
            pct = count/len(class_wrong)*100
            print(f"      → {pred_class:8s}: {count:2d} samples ({pct:5.1f}% of errors)")

# ==================== CONFUSION MATRIX ====================
print(f"\n" + "="*70)
print("📊 CONFUSION MATRIX")
print("="*70)

confusion = pd.crosstab(
    df['True_Class'], 
    df['Predicted_Class'], 
    rownames=['True'], 
    colnames=['Predicted'],
    margins=True
)
print(confusion)

# Tính metrics chi tiết
print(f"\n📈 PER-CLASS METRICS:")
print("-"*70)

for true_class in ['Health', 'Rust', 'Other']:
    tp = len(df[(df['True_Class'] == true_class) & (df['Predicted_Class'] == true_class)])
    fp = len(df[(df['True_Class'] != true_class) & (df['Predicted_Class'] == true_class)])
    fn = len(df[(df['True_Class'] == true_class) & (df['Predicted_Class'] != true_class)])
    tn = len(df[(df['True_Class'] != true_class) & (df['Predicted_Class'] != true_class)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{true_class}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

# ==================== TOP ERROR PATTERNS ====================
print(f"\n" + "="*70)
print("🔥 TOP ERROR PATTERNS (MOST COMMON MISTAKES)")
print("="*70)

wrong_df = df[df['Correct'] == 'No']
error_patterns = wrong_df.groupby(['True_Class', 'Predicted_Class']).size().sort_values(ascending=False)

print(f"\nRanking:")
for i, ((true_c, pred_c), count) in enumerate(error_patterns.items(), 1):
    pct = count / incorrect_count * 100
    print(f"  {i}. {true_c:8s} → {pred_c:8s}: {count:3d} samples ({pct:5.1f}% of all errors)")

# ==================== STEP 1 vs STEP 2 ANALYSIS ====================
print(f"\n" + "="*70)
print("🔍 STEP 1 (Healthy vs Diseased) ANALYSIS")
print("="*70)

# Step 1: Healthy vs Diseased
df['True_Step1'] = df['True_Label'].apply(lambda x: 'Healthy' if x == 0 else 'Diseased')
df['Pred_Step1'] = df['Predicted_Label'].apply(lambda x: 'Healthy' if x == 0 else 'Diseased')

step1_correct = (df['True_Step1'] == df['Pred_Step1']).sum()
step1_total = len(df)

print(f"\nStep 1 Accuracy: {step1_correct}/{step1_total} = {step1_correct/step1_total:.3f}")

# Step 1 confusion
step1_confusion = pd.crosstab(df['True_Step1'], df['Pred_Step1'], margins=True)
print(f"\nStep 1 Confusion Matrix:")
print(step1_confusion)

# Step 1 errors
print(f"\nStep 1 Errors:")
healthy_as_diseased = len(df[(df['True_Step1'] == 'Healthy') & (df['Pred_Step1'] == 'Diseased')])
diseased_as_healthy = len(df[(df['True_Step1'] == 'Diseased') & (df['Pred_Step1'] == 'Healthy')])

print(f"  Healthy → Diseased: {healthy_as_diseased} ({healthy_as_diseased/40*100:.1f}% of Healthy)")
print(f"  Diseased → Healthy: {diseased_as_healthy} ({diseased_as_healthy/80*100:.1f}% of Diseased)")

print(f"\n" + "="*70)
print("🔍 STEP 2 (Rust vs Other) ANALYSIS")
print("="*70)

# Step 2: Only for Diseased samples
diseased_df = df[df['True_Label'] > 0]

if len(diseased_df) > 0:
    step2_correct = len(diseased_df[diseased_df['Correct'] == 'Yes'])
    step2_total = len(diseased_df)
    
    print(f"\nStep 2 Accuracy (on diseased only): {step2_correct}/{step2_total} = {step2_correct/step2_total:.3f}")
    
    # Step 2 confusion (Rust vs Other)
    diseased_true = diseased_df['True_Class'].replace({'Rust': 'Rust', 'Other': 'Other'})
    diseased_pred = diseased_df['Predicted_Class'].replace({'Rust': 'Rust', 'Other': 'Other'})
    step2_confusion = pd.crosstab(diseased_true, diseased_pred, margins=True)
    print(f"\nStep 2 Confusion Matrix:")
    print(step2_confusion)

# ==================== DANH SÁCH FILE SAI ====================
print(f"\n" + "="*70)
print("📁 DANH SÁCH CÁC FILE BỊ DỰ ĐOÁN SAI")
print("="*70)

print(f"\n🔴 Health bị nhầm thành Rust ({len(df[(df['True_Class'] == 'Health') & (df['Predicted_Class'] == 'Rust')])} files):")
health_as_rust = wrong_df[(wrong_df['True_Class'] == 'Health') & (wrong_df['Predicted_Class'] == 'Rust')]
for idx, row in health_as_rust.iterrows():
    print(f"    - {row['Image_Path']}")

print(f"\n🔴 Health bị nhầm thành Other ({len(df[(df['True_Class'] == 'Health') & (df['Predicted_Class'] == 'Other')])} files):")
health_as_other = wrong_df[(wrong_df['True_Class'] == 'Health') & (wrong_df['Predicted_Class'] == 'Other')]
for idx, row in health_as_other.iterrows():
    print(f"    - {row['Image_Path']}")

print(f"\n🔴 Rust bị nhầm thành Health ({len(df[(df['True_Class'] == 'Rust') & (df['Predicted_Class'] == 'Health')])} files):")
rust_as_health = wrong_df[(wrong_df['True_Class'] == 'Rust') & (wrong_df['Predicted_Class'] == 'Health')]
for idx, row in rust_as_health.iterrows():
    print(f"    - {row['Image_Path']}")

print(f"\n🔴 Rust bị nhầm thành Other ({len(df[(df['True_Class'] == 'Rust') & (df['Predicted_Class'] == 'Other')])} files):")
rust_as_other = wrong_df[(wrong_df['True_Class'] == 'Rust') & (wrong_df['Predicted_Class'] == 'Other')]
for idx, row in rust_as_other.iterrows():
    print(f"    - {row['Image_Path']}")

print(f"\n🔴 Other bị nhầm thành Health ({len(df[(df['True_Class'] == 'Other') & (df['Predicted_Class'] == 'Health')])} files):")
other_as_health = wrong_df[(wrong_df['True_Class'] == 'Other') & (wrong_df['Predicted_Class'] == 'Health')]
for idx, row in other_as_health.iterrows():
    print(f"    - {row['Image_Path']}")

print(f"\n🔴 Other bị nhầm thành Rust ({len(df[(df['True_Class'] == 'Other') & (df['Predicted_Class'] == 'Rust')])} files):")
other_as_rust = wrong_df[(wrong_df['True_Class'] == 'Other') & (wrong_df['Predicted_Class'] == 'Rust')]
for idx, row in other_as_rust.iterrows():
    print(f"    - {row['Image_Path']}")

# ==================== KHUYẾN NGHỊ CẢI THIỆN ====================
print(f"\n" + "="*70)
print("💡 KHUYẾN NGHỊ CẢI THIỆN MODEL")
print("="*70)

# Phân tích để đưa ra khuyến nghị
healthy_recall = len(df[(df['True_Class'] == 'Health') & (df['Predicted_Class'] == 'Health')]) / len(df[df['True_Class'] == 'Health'])
diseased_as_healthy_pct = diseased_as_healthy / 80 * 100

print(f"\n🎯 Ưu tiên cải thiện:")

if healthy_recall < 0.6:
    print(f"\n1️⃣ CRITICAL: Healthy Recall = {healthy_recall:.1%} (< 60%)")
    print(f"   ❌ Vấn đề: {healthy_as_diseased}/40 Healthy bị nhầm Diseased")
    print(f"   💊 Giải pháp:")
    print(f"      - Tăng Focal Loss alpha: 3.0 → 4.5")
    print(f"      - Tăng Oversample: 2x → 4x")
    print(f"      - Add MixUp augmentation cho Healthy")
    print(f"      - Tune threshold: 0.5 → 0.3-0.35")

if diseased_as_healthy_pct > 15:
    print(f"\n2️⃣ HIGH: Diseased→Healthy = {diseased_as_healthy_pct:.1f}% (> 15%)")
    print(f"   ❌ Vấn đề: {diseased_as_healthy}/80 cây bệnh bị bỏ sót")
    print(f"   💊 Giải pháp:")
    print(f"      - Tăng RGB handcrafted features (thêm Gabor, CCV)")
    print(f"      - Tăng spectral attention heads: 4 → 8")
    print(f"      - Add hard negative mining")

# Step 2 analysis
if len(diseased_df) > 0:
    rust_recall = len(diseased_df[(diseased_df['True_Class'] == 'Rust') & (diseased_df['Predicted_Class'] == 'Rust')]) / len(diseased_df[diseased_df['True_Class'] == 'Rust'])
    other_recall = len(diseased_df[(diseased_df['True_Class'] == 'Other') & (diseased_df['Predicted_Class'] == 'Other')]) / len(diseased_df[diseased_df['True_Class'] == 'Other'])
    
    if rust_recall < 0.7 or other_recall < 0.7:
        print(f"\n3️⃣ MEDIUM: Step 2 Performance")
        print(f"   Rust Recall: {rust_recall:.1%}")
        print(f"   Other Recall: {other_recall:.1%}")
        print(f"   💊 Giải pháp:")
        print(f"      - Tăng Spectral weight trong Step 2: 70% → 80%")
        print(f"      - Add spectral augmentation (SpecAugment)")
        print(f"      - Tăng HS encoder depth")

print(f"\n4️⃣ ADVANCED TECHNIQUES:")
print(f"   - Test-Time Augmentation (TTA): +2-4% accuracy")
print(f"   - Ensemble 3 models: +4-6% accuracy")
print(f"   - Pseudo-labeling unlabeled data")

print("\n" + "="*70)
print("✅ PHÂN TÍCH HOÀN TẤT!")
print("="*70)

# Save error report to file
output_file = Path('error_analysis_report.txt')
print(f"\n💾 Saving detailed report to: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ERROR ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Total samples: {total}\n")
    f.write(f"Accuracy: {correct_count/total:.3f}\n\n")
    
    f.write("Files with errors:\n")
    f.write("-"*70 + "\n")
    for idx, row in wrong_df.iterrows():
        f.write(f"{row['Image_Path']}: True={row['True_Class']}, Pred={row['Predicted_Class']}\n")

print(f"✅ Report saved successfully!\n")
