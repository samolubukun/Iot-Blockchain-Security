# IoT Blockchain Security Threat Detection - Complete ML System
# Final Year Thesis - Comparative Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("IoT BLOCKCHAIN SECURITY - ML COMPARATIVE ANALYSIS SYSTEM")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[STEP 1] Loading and preprocessing data...")

# Load data
df = pd.read_csv('iot_blockchain_security_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")

# Display basic info
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Threat Mitigated'].value_counts()}")

# Encode categorical variables
le_layer = LabelEncoder()
le_request = LabelEncoder()
le_threat = LabelEncoder()
le_consensus = LabelEncoder()

df['IoT Layer Encoded'] = le_layer.fit_transform(df['IoT Layer'])
df['Request Type Encoded'] = le_request.fit_transform(df['Request Type'])
df['Security Threat Type Encoded'] = le_threat.fit_transform(df['Security Threat Type'])
df['Consensus Mechanism Encoded'] = le_consensus.fit_transform(df['Consensus Mechanism'])

# Feature selection
feature_cols = ['IoT Layer Encoded', 'Request Type Encoded', 'Data Size (KB)', 
                'Processing Time (ms)', 'Security Threat Type Encoded', 
                'Attack Severity (0-10)', 'Blockchain Transaction Time (ms)', 
                'Consensus Mechanism Encoded', 'Energy Consumption (mJ)']

X = df[feature_cols]
y = df['Threat Mitigated']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'layer': le_layer, 'request': le_request, 'threat': le_threat, 'consensus': le_consensus}, 
            'label_encoders.pkl')
print("\n✓ Preprocessing objects saved")

# ============================================================================
# 2. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "="*80)
print("[STEP 2] Training and evaluating models...")
print("="*80)

models = {}
results = {}

# Model 1: Random Forest
print("\n[1/5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12, min_samples_split=8, 
                            min_samples_leaf=4, class_weight='balanced', max_features='sqrt')
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf
print("✓ Random Forest trained")

# Model 2: Logistic Regression
print("\n[2/5] Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=2000, solver='saga', class_weight='balanced', 
                        C=0.5, penalty='l2', multi_class='multinomial')
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr
print("✓ Logistic Regression trained")

# Model 3: Decision Tree
print("\n[3/5] Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5, 
                            class_weight='balanced', criterion='gini')
dt.fit(X_train_scaled, y_train)
models['Decision Tree'] = dt
print("✓ Decision Tree trained")

# Model 4: Gradient Boosting
print("\n[4/5] Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.08, max_depth=4, 
                                min_samples_split=10, min_samples_leaf=5, subsample=0.8, max_features='sqrt')
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb
print("✓ Gradient Boosting trained")

# Model 5: Isolation Forest (for anomaly detection)
print("\n[5/5] Training Isolation Forest...")
# Use the actual class distribution to set contamination
contamination_rate = (y_train == 0).sum() / len(y_train)  # Proportion of threats (not mitigated)
iso = IsolationForest(contamination=min(contamination_rate * 1.2, 0.3), random_state=42, n_estimators=120)
iso.fit(X_train_scaled)
# Convert predictions: -1 (anomaly/threat) -> 0, 1 (normal/mitigated) -> 1
iso_pred_train = np.where(iso.predict(X_train_scaled) == 1, 1, 0)
iso_pred_test = np.where(iso.predict(X_test_scaled) == 1, 1, 0)
models['Isolation Forest'] = iso
print("✓ Isolation Forest trained")

# ============================================================================
# 3. COMPREHENSIVE EVALUATION
# ============================================================================
print("\n" + "="*80)
print("[STEP 3] Evaluating models on test set...")
print("="*80)

for name, model in models.items():
    if name == 'Isolation Forest':
        y_pred = iso_pred_test
        y_pred_proba = None
    else:
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC-AUC
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        # For Isolation Forest, use score_samples
        if hasattr(model, 'score_samples'):
            anomaly_scores = model.score_samples(X_test_scaled)
            roc_auc = roc_auc_score(y_test, anomaly_scores)
        else:
            roc_auc = None
    
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 4. VISUALIZATION - COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[STEP 4] Generating visualizations...")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 24))

# 4.1: Model Performance Comparison
ax1 = plt.subplot(4, 3, 1)
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})
metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax1)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim([0, 1.1])
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4.2: Confusion Matrices for all models
for idx, (name, model) in enumerate(models.items(), start=2):
    ax = plt.subplot(4, 3, idx)
    cm = confusion_matrix(y_test, results[name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# 4.3: ROC Curves
ax7 = plt.subplot(4, 3, 7)
for name in models.keys():
    if results[name]['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        roc_auc = auc(fpr, tpr)
        ax7.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
ax7.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax7.set_xlabel('False Positive Rate')
ax7.set_ylabel('True Positive Rate')
ax7.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax7.legend(loc='lower right')
ax7.grid(True, alpha=0.3)

# 4.4: Precision-Recall Curves
ax8 = plt.subplot(4, 3, 8)
for name in models.keys():
    if results[name]['probabilities'] is not None:
        precision, recall, _ = precision_recall_curve(y_test, results[name]['probabilities'])
        ax8.plot(recall, precision, label=name, linewidth=2)
ax8.set_xlabel('Recall')
ax8.set_ylabel('Precision')
ax8.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax8.legend(loc='lower left')
ax8.grid(True, alpha=0.3)

# 4.5: Feature Importance (Random Forest)
ax9 = plt.subplot(4, 3, 9)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)
ax9.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
ax9.set_xlabel('Importance Score')
ax9.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='x')

# 4.6: Target Distribution
ax10 = plt.subplot(4, 3, 10)
threat_counts = y.value_counts()
ax10.pie(threat_counts, labels=['Mitigated', 'Not Mitigated'], autopct='%1.1f%%', 
         colors=['#2ecc71', '#e74c3c'], startangle=90)
ax10.set_title('Target Class Distribution', fontsize=14, fontweight='bold')

# 4.7: Attack Severity Distribution
ax11 = plt.subplot(4, 3, 11)
ax11.hist(df['Attack Severity (0-10)'], bins=11, color='coral', edgecolor='black', alpha=0.7)
ax11.set_xlabel('Attack Severity')
ax11.set_ylabel('Frequency')
ax11.set_title('Attack Severity Distribution', fontsize=14, fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

# 4.8: Consensus Mechanism Performance
ax12 = plt.subplot(4, 3, 12)
consensus_threat = df.groupby('Consensus Mechanism')['Threat Mitigated'].mean().sort_values()
ax12.barh(consensus_threat.index, consensus_threat.values, color='lightgreen')
ax12.set_xlabel('Threat Mitigation Rate')
ax12.set_title('Threat Mitigation by Consensus Mechanism', fontsize=14, fontweight='bold')
ax12.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Main analysis plot saved: comprehensive_model_analysis.png")
plt.close()

# ============================================================================
# 5. ADDITIONAL DETAILED VISUALIZATIONS
# ============================================================================

# 5.1: Classification Reports (Text-based visualizations)
fig2 = plt.figure(figsize=(20, 15))
for idx, name in enumerate(models.keys(), start=1):
    ax = plt.subplot(3, 2, idx)
    report = classification_report(y_test, results[name]['predictions'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-3, :3]  # Exclude accuracy, macro, weighted avg
    
    sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title(f'{name} - Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Class')
    ax.set_xlabel('Metric')

plt.tight_layout()
plt.savefig('classification_reports_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Classification reports saved: classification_reports_heatmap.png")
plt.close()

# 5.2: Error Analysis
fig3 = plt.figure(figsize=(20, 12))

# Best model error analysis (Random Forest typically performs best)
best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
best_predictions = results[best_model_name]['predictions']

# Misclassification by feature
misclassified_idx = y_test != best_predictions
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

ax1 = plt.subplot(2, 3, 1)
correct_severity = df.loc[X_test[~misclassified_idx].index, 'Attack Severity (0-10)']
wrong_severity = df.loc[X_test[misclassified_idx].index, 'Attack Severity (0-10)']
ax1.hist([correct_severity, wrong_severity], bins=11, label=['Correct', 'Misclassified'], 
         color=['green', 'red'], alpha=0.6)
ax1.set_xlabel('Attack Severity')
ax1.set_ylabel('Count')
ax1.set_title('Misclassification by Attack Severity', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
threat_types = df.loc[X_test[misclassified_idx].index, 'Security Threat Type'].value_counts()
ax2.bar(threat_types.index, threat_types.values, color='salmon')
ax2.set_xlabel('Threat Type')
ax2.set_ylabel('Misclassification Count')
ax2.set_title('Misclassifications by Threat Type', fontsize=12, fontweight='bold')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

ax3 = plt.subplot(2, 3, 3)
layer_misclass = df.loc[X_test[misclassified_idx].index, 'IoT Layer'].value_counts()
ax3.bar(layer_misclass.index, layer_misclass.values, color='lightcoral')
ax3.set_xlabel('IoT Layer')
ax3.set_ylabel('Misclassification Count')
ax3.set_title('Misclassifications by IoT Layer', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Correlation heatmap
ax4 = plt.subplot(2, 3, 4)
correlation_matrix = df[feature_cols + ['Threat Mitigated']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, 
            cbar_kws={'label': 'Correlation'}, center=0)
ax4.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# Processing time vs mitigation
ax5 = plt.subplot(2, 3, 5)
mitigated = df[df['Threat Mitigated'] == 1]['Processing Time (ms)']
not_mitigated = df[df['Threat Mitigated'] == 0]['Processing Time (ms)']
ax5.boxplot([mitigated, not_mitigated], labels=['Mitigated', 'Not Mitigated'])
ax5.set_ylabel('Processing Time (ms)')
ax5.set_title('Processing Time by Mitigation Status', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Energy consumption analysis
ax6 = plt.subplot(2, 3, 6)
mitigated_energy = df[df['Threat Mitigated'] == 1]['Energy Consumption (mJ)']
not_mitigated_energy = df[df['Threat Mitigated'] == 0]['Energy Consumption (mJ)']
ax6.boxplot([mitigated_energy, not_mitigated_energy], labels=['Mitigated', 'Not Mitigated'])
ax6.set_ylabel('Energy Consumption (mJ)')
ax6.set_title('Energy Consumption by Mitigation Status', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_analysis_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Error analysis saved: error_analysis_detailed.png")
plt.close()

# 5.3: Model Comparison Summary Table
fig4, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

summary_data = []
for name in results.keys():
    roc_auc_val = results[name]['roc_auc'] if results[name]['roc_auc'] is not None else 0.0
    summary_data.append([
        name,
        f"{results[name]['accuracy']:.4f}",
        f"{results[name]['precision']:.4f}",
        f"{results[name]['recall']:.4f}",
        f"{results[name]['f1_score']:.4f}",
        f"{roc_auc_val:.4f}"
    ])

table = ax.table(cellText=summary_data, 
                colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                cellLoc='center', loc='center', 
                colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Color code the best scores
for i in range(1, 6):
    scores = [float(summary_data[j][i]) for j in range(len(summary_data))]
    best_idx = scores.index(max(scores))
    table[(best_idx + 1, i)].set_facecolor('#90EE90')

ax.set_title('Model Performance Summary - All Metrics', fontsize=16, fontweight='bold', pad=20)
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Comparison table saved: model_comparison_table.png")
plt.close()

# ============================================================================
# 6. SELECT AND SAVE BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("[STEP 5] Selecting and saving best model...")
print("="*80)

best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
best_model = models[best_model_name]
best_score = results[best_model_name]['f1_score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {best_score:.4f}")
print(f"\n   Detailed Metrics:")
print(f"   - Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"   - Precision: {results[best_model_name]['precision']:.4f}")
print(f"   - Recall:    {results[best_model_name]['recall']:.4f}")

# Save best model
joblib.dump(best_model, f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
print(f"\n✓ Best model saved: best_model_{best_model_name.replace(' ', '_').lower()}.pkl")

# Save all models
for name, model in models.items():
    joblib.dump(model, f'model_{name.replace(" ", "_").lower()}.pkl')
print("✓ All models saved individually")

# ============================================================================
# 7. GENERATE FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("[STEP 6] Generating final report...")
print("="*80)

with open('model_performance_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("IoT BLOCKCHAIN SECURITY - MACHINE LEARNING ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Train/Test split: {len(X_train)}/{len(X_test)}\n")
    f.write(f"Target distribution: {dict(y.value_counts())}\n\n")
    
    f.write("MODELS EVALUATED\n")
    f.write("-"*80 + "\n")
    for idx, name in enumerate(models.keys(), 1):
        f.write(f"{idx}. {name}\n")
    f.write("\n")
    
    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-"*80 + "\n")
    for name in results.keys():
        f.write(f"{name:<25} {results[name]['accuracy']:<12.4f} "
                f"{results[name]['precision']:<12.4f} {results[name]['recall']:<12.4f} "
                f"{results[name]['f1_score']:<12.4f}\n")
    f.write("\n")
    
    f.write("BEST MODEL SELECTION\n")
    f.write("-"*80 + "\n")
    f.write(f"Selected Model: {best_model_name}\n")
    f.write(f"F1-Score: {best_score:.4f}\n")
    f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
    f.write(f"Precision: {results[best_model_name]['precision']:.4f}\n")
    f.write(f"Recall: {results[best_model_name]['recall']:.4f}\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORT (BEST MODEL)\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test, results[best_model_name]['predictions']))
    f.write("\n")
    
    f.write("FILES GENERATED\n")
    f.write("-"*80 + "\n")
    f.write("1. best_model_*.pkl - Best performing model\n")
    f.write("2. scaler.pkl - Feature scaler\n")
    f.write("3. label_encoders.pkl - Categorical encoders\n")
    f.write("4. comprehensive_model_analysis.png - Main analysis plots\n")
    f.write("5. classification_reports_heatmap.png - Detailed metrics\n")
    f.write("6. error_analysis_detailed.png - Error analysis\n")
    f.write("7. model_comparison_table.png - Performance summary\n")
    f.write("8. model_performance_report.txt - This report\n")

print("✓ Report saved: model_performance_report.txt")

# ============================================================================
# 8. DISPLAY COMPREHENSIVE RESULTS IN TERMINAL
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL EVALUATION RESULTS")
print("="*80)

print("\n" + "-"*80)
print("SUMMARY TABLE")
print("-"*80)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-"*80)
for name in results.keys():
    roc_auc_val = results[name]['roc_auc'] if results[name]['roc_auc'] is not None else 0.0
    print(f"{name:<25} {results[name]['accuracy']:<12.4f} "
          f"{results[name]['precision']:<12.4f} {results[name]['recall']:<12.4f} "
          f"{results[name]['f1_score']:<12.4f} {roc_auc_val:<12.4f}")
print("-"*80)

print("\n" + "-"*80)
print("DETAILED RESULTS FOR ALL MODELS")
print("-"*80)

for name in models.keys():
    print(f"\n{'='*80}")
    print(f"MODEL: {name.upper()}")
    print(f"{'='*80}")
    
    print(f"\nPerformance Metrics:")
    print(f"  • Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  • Precision: {results[name]['precision']:.4f}")
    print(f"  • Recall:    {results[name]['recall']:.4f}")
    print(f"  • F1-Score:  {results[name]['f1_score']:.4f}")
    if results[name]['roc_auc'] is not None:
        print(f"  • ROC-AUC:   {results[name]['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, results[name]['predictions'])
    print(f"                Predicted")
    print(f"                Threat Active (0)  Threat Mitigated (1)")
    print(f"Actual Threat Active (0)     {cm[0, 0]:<18}  {cm[0, 1]}")
    print(f"Actual Threat Mitigated (1)  {cm[1, 0]:<18}  {cm[1, 1]}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  • True Negatives (TN):      {tn}")
    print(f"  • True Positives (TP):      {tp}")
    print(f"  • False Positives (FP):     {fp}")
    print(f"  • False Negatives (FN):     {fn}")
    print(f"  • Specificity (TN/TN+FP):   {tn/(tn+fp):.4f}")
    print(f"  • Sensitivity (TP/TP+FN):   {tp/(tp+fn):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, results[name]['predictions'], 
                              target_names=['Threat Active', 'Threat Mitigated']))

print("\n" + "="*80)

# ============================================================================
# 9. SAVE MODEL METADATA
# ============================================================================
model_metadata = {
    'best_model': best_model_name,
    'feature_columns': feature_cols,
    'model_performance': results,
    'label_encoders': {
        'layer_classes': list(le_layer.classes_),
        'request_classes': list(le_request.classes_),
        'threat_classes': list(le_threat.classes_),
        'consensus_classes': list(le_consensus.classes_)
    }
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("✓ Model metadata saved: model_metadata.pkl")

# ============================================================================
# 10. COMPREHENSIVE CLASSIFICATION REPORTS & ALL-MODELS VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("[STEP 9] Generating comprehensive reports for all models...")
print("="*80)

# 9.1: Generate comprehensive classification report file
with open('comprehensive_classification_reports.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("IoT BLOCKCHAIN SECURITY - COMPREHENSIVE CLASSIFICATION REPORTS\n")
    f.write("ALL MODELS COMPARISON\n")
    f.write("="*80 + "\n\n")
    
    f.write("MODEL OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Models Evaluated: {len(models)}\n")
    f.write(f"Test Set Size: {len(y_test)}\n")
    f.write(f"Class Distribution (Test): {dict(y_test.value_counts())}\n\n")
    
    f.write("PERFORMANCE SUMMARY\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-"*80 + "\n")
    for name in results.keys():
        f.write(f"{name:<25} {results[name]['accuracy']:<12.4f} "
                f"{results[name]['precision']:<12.4f} {results[name]['recall']:<12.4f} "
                f"{results[name]['f1_score']:<12.4f}\n")
    f.write("\n\n")
    
    # Detailed reports for each model
    for name in models.keys():
        f.write("="*80 + "\n")
        f.write(f"MODEL: {name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        cm = confusion_matrix(y_test, results[name]['predictions'])
        f.write(f"                Predicted\n")
        f.write(f"                Threat Active (0)  Threat Mitigated (1)\n")
        f.write(f"Actual Threat Active (0)     {cm[0, 0]:<18}  {cm[0, 1]}\n")
        f.write(f"Actual Threat Mitigated (1)  {cm[1, 0]:<18}  {cm[1, 1]}\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test, results[name]['predictions'], 
                                     target_names=['Threat Active', 'Threat Mitigated']))
        f.write("\n")
        
        f.write("DETAILED METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:  {results[name]['accuracy']:.4f}\n")
        f.write(f"Precision: {results[name]['precision']:.4f}\n")
        f.write(f"Recall:    {results[name]['recall']:.4f}\n")
        f.write(f"F1-Score:  {results[name]['f1_score']:.4f}\n\n")
        
        # True Positives, True Negatives, False Positives, False Negatives
        tn, fp, fn, tp = cm.ravel()
        f.write("CONFUSION MATRIX BREAKDOWN\n")
        f.write("-"*80 + "\n")
        f.write(f"True Negatives (Correctly identified threats):      {tn}\n")
        f.write(f"True Positives (Correctly identified mitigated):    {tp}\n")
        f.write(f"False Positives (Incorrectly identified mitigated): {fp}\n")
        f.write(f"False Negatives (Missed threats):                   {fn}\n")
        f.write(f"Specificity (True Negative Rate):                   {tn/(tn+fp):.4f}\n")
        f.write(f"Sensitivity (True Positive Rate):                   {tp/(tp+fn):.4f}\n\n\n")

print("✓ Comprehensive reports saved: comprehensive_classification_reports.txt")

# 9.2: ROC-AUC Curves for all models
fig_roc = plt.figure(figsize=(14, 8))
ax_roc = fig_roc.add_subplot(111)

colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for idx, (name, color) in enumerate(zip(models.keys(), colors)):
    if results[name]['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                   linewidth=2.5, color=color)
    else:
        # For Isolation Forest, we can use score_samples
        if hasattr(models[name], 'score_samples'):
            anomaly_scores = models[name].score_samples(X_test_scaled)
            fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                       linewidth=2.5, color=color, linestyle='--')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.500)')
ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax_roc.set_title('ROC-AUC Curves - All Models Comparison', fontsize=14, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=11)
ax_roc.grid(True, alpha=0.3)
ax_roc.set_xlim([-0.02, 1.02])
ax_roc.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('roc_auc_all_models.png', dpi=300, bbox_inches='tight')
print("✓ ROC-AUC curves saved: roc_auc_all_models.png")
plt.close()

# 9.3: Precision-Recall Curves for all models
fig_pr = plt.figure(figsize=(14, 8))
ax_pr = fig_pr.add_subplot(111)

for idx, (name, color) in enumerate(zip(models.keys(), colors)):
    if results[name]['probabilities'] is not None:
        precision, recall, _ = precision_recall_curve(y_test, results[name]['probabilities'])
        ap = np.mean(precision)
        ax_pr.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', 
                  linewidth=2.5, color=color)
    else:
        if hasattr(models[name], 'score_samples'):
            anomaly_scores = models[name].score_samples(X_test_scaled)
            precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
            ap = np.mean(precision)
            ax_pr.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', 
                      linewidth=2.5, color=color, linestyle='--')

ax_pr.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax_pr.set_title('Precision-Recall Curves - All Models Comparison', fontsize=14, fontweight='bold')
ax_pr.legend(loc='lower left', fontsize=11)
ax_pr.grid(True, alpha=0.3)
ax_pr.set_xlim([-0.02, 1.02])
ax_pr.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('precision_recall_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Precision-Recall curves saved: precision_recall_all_models.png")
plt.close()

# 9.4: Confusion Matrices for all models in one figure
fig_cm = plt.figure(figsize=(16, 10))

for idx, (name, model) in enumerate(models.items(), start=1):
    ax = plt.subplot(2, 3, idx)
    cm = confusion_matrix(y_test, results[name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax, cbar=True,
                xticklabels=['Threat Active', 'Threat Mitigated'],
                yticklabels=['Threat Active', 'Threat Mitigated'])
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved: confusion_matrices_all_models.png")
plt.close()

# 9.5: Performance metrics comparison (bar plot)
fig_metrics = plt.figure(figsize=(14, 8))

metrics_comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})

x = np.arange(len(metrics_comparison))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - 1.5*width, metrics_comparison['Accuracy'], width, label='Accuracy', color='#3498db')
ax.bar(x - 0.5*width, metrics_comparison['Precision'], width, label='Precision', color='#2ecc71')
ax.bar(x + 0.5*width, metrics_comparison['Recall'], width, label='Recall', color='#e74c3c')
ax.bar(x + 1.5*width, metrics_comparison['F1-Score'], width, label='F1-Score', color='#f39c12')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics Comparison - All Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison['Model'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metrics_comparison_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Metrics comparison saved: metrics_comparison_all_models.png")
plt.close()

# 9.6: Model ranking by F1-Score
fig_rank = plt.figure(figsize=(12, 7))

ranking = pd.DataFrame({
    'Model': list(results.keys()),
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
}).sort_values('F1-Score', ascending=True)

colors_rank = ['#2ecc71' if i == len(ranking) - 1 else '#3498db' for i in range(len(ranking))]
ax = fig_rank.add_subplot(111)
ax.barh(ranking['Model'], ranking['F1-Score'], color=colors_rank)

for i, (model, score) in enumerate(zip(ranking['Model'], ranking['F1-Score'])):
    ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontweight='bold')

ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Model Ranking by F1-Score', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.0])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_ranking_f1score.png', dpi=300, bbox_inches='tight')
print("✓ Model ranking saved: model_ranking_f1score.png")
plt.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nSUMMARY:")
print(f"• Best Model: {best_model_name} (F1-Score: {best_score:.4f})")
print(f"• Total Models Trained: {len(models)}")
print(f"• Visualizations Generated: 9 comprehensive plots")
print(f"• Models Saved: {len(models) + 1} (including best model)")
print("\nFILES GENERATED:")
print("  Reports:")
print("  • model_performance_report.txt")
print("  • comprehensive_classification_reports.txt")
print("  ")
print("  Visualizations:")
print("  • roc_auc_all_models.png")
print("  • precision_recall_all_models.png")
print("  • confusion_matrices_all_models.png")
print("  • metrics_comparison_all_models.png")
print("  • model_ranking_f1score.png")
print("  • comprehensive_model_analysis.png")
print("  • classification_reports_heatmap.png")
print("  • error_analysis_detailed.png")
print("  • model_comparison_table.png")
print("\nREADY FOR DEPLOYMENT!")
print("Use the best model file for Streamlit or other interfaces.")
print("="*80)