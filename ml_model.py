import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier  # Added
from sklearn.impute import SimpleImputer  # Added
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import re
warnings.filterwarnings('ignore')

# ============================================
# 1. Directory Setup
print("=" * 60)
print("Arabic Handwriting Quality Classification System using RandomForest")
print("=" * 60)

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
GOOD_DIR = os.path.join(current_dir, "GOOD")
BAD_DIR = os.path.join(current_dir, "BAD")
MODELS_DIR = os.path.join(current_dir, "models")
RESULTS_DIR = os.path.join(current_dir, "results")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f" Current directory: {current_dir}")
print(f" GOOD: {GOOD_DIR}")
print(f" BAD: {BAD_DIR}")

# 2. Check Available Data
# ============================================
print("\n Checking available data...")

# Check if directories exist
if not os.path.exists(GOOD_DIR):
    print(f"ERROR: Directory '{GOOD_DIR}' not found!")
    print("   Make sure the GOOD folder is in the same location as main.py")
    exit(1)

if not os.path.exists(BAD_DIR):
    print(f"ERROR: Directory '{BAD_DIR}' not found!")
    print("   Make sure the BAD folder is in the same location as main.py")
    exit(1)

# Count files
good_files = [f for f in os.listdir(GOOD_DIR) if f.endswith(".csv")]
bad_files = [f for f in os.listdir(BAD_DIR) if f.endswith(".csv")]

print(f"GOOD: {len(good_files)} CSV files")
print(f"BAD: {len(bad_files)} CSV files")

if len(good_files) == 0:
    print("  Warning: GOOD folder is empty! Use server.py to collect good data")
if len(bad_files) == 0:
    print("  Warning: BAD folder is empty! Use server.py to collect bad data")

if len(good_files) == 0 or len(bad_files) == 0:
    print(" Cannot train without sufficient data")
    exit(1)

# ============================================
# 3. Enhanced File Reading Function
# ============================================
def read_csv_safe(file_path):
    """
    Read CSV file safely with error handling
    """
    try:
        # Try to read the file normally
        df = pd.read_csv(file_path)
        
        # Check for essential columns
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        # If some columns are missing
        missing_cols = [col for col in sensor_cols if col not in df.columns]
        if missing_cols:
            print(f"     Missing columns in {os.path.basename(file_path)}: {missing_cols}")
            # Add missing columns as NaN
            for col in missing_cols:
                df[col] = np.nan
        
        # Convert numeric columns
        for col in sensor_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"     Error reading {os.path.basename(file_path)}: {e}")
        
        # Try reading as raw text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Search for numeric data
            data = []
            for line in lines:
                # Extract all numbers from the line
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if len(numbers) >= 6:  # We need at least 6 values
                    # Take first 6 numbers (ax, ay, az, gx, gy, gz)
                    row_data = numbers[:6]
                    # If there are more, take timestamp and type
                    if len(numbers) >= 7:
                        row_data = [numbers[0]] + row_data  # timestamp first
                    if len(numbers) >= 8:
                        row_data = row_data + [numbers[7]]  # type last
                    
                    # If we have 8 values (timestamp, 6 sensors, type)
                    if len(row_data) == 8:
                        data.append(row_data)
            
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'type'])
                
                # Convert numeric columns
                numeric_cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                print(f"   Fixed {os.path.basename(file_path)} (from raw text)")
                return df
            else:
                print(f"    Cannot fix {os.path.basename(file_path)}")
                return None
                
        except Exception as e2:
            print(f"    Failed to fix {os.path.basename(file_path)}: {e2}")
            return None

# ============================================
# 4. Feature Extraction (Updated)
# ============================================
def extract_features(df):
    """Extract statistical features from sensor data"""
    features = {}
    
    # Sensor columns
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for col in sensor_cols:
        if col in df.columns:
            # Convert to numeric with error handling
            col_data = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate features with NaN handling
            features[f'{col}_mean'] = np.nanmean(col_data) if not col_data.isna().all() else 0
            features[f'{col}_std'] = np.nanstd(col_data) if not col_data.isna().all() else 0.0001  # Avoid zero
            features[f'{col}_max'] = np.nanmax(col_data) if not col_data.isna().all() else 0
            features[f'{col}_min'] = np.nanmin(col_data) if not col_data.isna().all() else 0
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            features[f'{col}_median'] = np.nanmedian(col_data) if not col_data.isna().all() else 0
            features[f'{col}_skew'] = col_data.skew() if not col_data.isna().all() and col_data.nunique() > 1 else 0
            features[f'{col}_kurtosis'] = col_data.kurtosis() if not col_data.isna().all() and col_data.nunique() > 1 else 0
            features[f'{col}_energy'] = np.nansum(col_data ** 2)
            features[f'{col}_abs_mean'] = np.nanmean(np.abs(col_data)) if not col_data.isna().all() else 0
            
            # Rate of change
            if len(col_data) > 1:
                diff = col_data.diff().dropna()
                if len(diff) > 0:
                    features[f'{col}_diff_mean'] = np.nanmean(diff) if not diff.isna().all() else 0
                    features[f'{col}_diff_std'] = np.nanstd(diff) if not diff.isna().all() and len(diff) > 1 else 0.0001
                else:
                    features[f'{col}_diff_mean'] = 0
                    features[f'{col}_diff_std'] = 0
            else:
                features[f'{col}_diff_mean'] = 0
                features[f'{col}_diff_std'] = 0
    
    # Combined features
    if all(col in df.columns for col in ['ax', 'ay', 'az']):
        # Use values without NaN for calculation
        ax_data = pd.to_numeric(df['ax'], errors='coerce').dropna()
        ay_data = pd.to_numeric(df['ay'], errors='coerce').dropna()
        az_data = pd.to_numeric(df['az'], errors='coerce').dropna()
        
        if len(ax_data) > 0 and len(ay_data) > 0 and len(az_data) > 0:
            min_len = min(len(ax_data), len(ay_data), len(az_data))
            acc_magnitude = np.sqrt(ax_data.iloc[:min_len]**2 + ay_data.iloc[:min_len]**2 + az_data.iloc[:min_len]**2)
            features['acc_magnitude_mean'] = np.mean(acc_magnitude)
            features['acc_magnitude_std'] = np.std(acc_magnitude) if len(acc_magnitude) > 1 else 0.0001
            features['acc_magnitude_max'] = np.max(acc_magnitude)
        else:
            features['acc_magnitude_mean'] = 0
            features['acc_magnitude_std'] = 0
            features['acc_magnitude_max'] = 0
    
    if all(col in df.columns for col in ['gx', 'gy', 'gz']):
        gx_data = pd.to_numeric(df['gx'], errors='coerce').dropna()
        gy_data = pd.to_numeric(df['gy'], errors='coerce').dropna()
        gz_data = pd.to_numeric(df['gz'], errors='coerce').dropna()
        
        if len(gx_data) > 0 and len(gy_data) > 0 and len(gz_data) > 0:
            min_len = min(len(gx_data), len(gy_data), len(gz_data))
            gyro_magnitude = np.sqrt(gx_data.iloc[:min_len]**2 + gy_data.iloc[:min_len]**2 + gz_data.iloc[:min_len]**2)
            features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
            features['gyro_magnitude_std'] = np.std(gyro_magnitude) if len(gyro_magnitude) > 1 else 0.0001
            features['gyro_magnitude_max'] = np.max(gyro_magnitude)
        else:
            features['gyro_magnitude_mean'] = 0
            features['gyro_magnitude_std'] = 0
            features['gyro_magnitude_max'] = 0
    
    # Type feature (Word/Sentence)
    if 'type' in df.columns:
        type_val = df['type'].iloc[0] if not df['type'].empty else 'Word'
        features['is_sentence'] = 1 if isinstance(type_val, str) and 'Sentence' in type_val else 0
    else:
        features['is_sentence'] = 0
    
    # Time features
    if 'timestamp' in df.columns:
        timestamp_data = pd.to_numeric(df['timestamp'], errors='coerce').dropna()
        if len(timestamp_data) > 1:
            time_diff = timestamp_data.diff().dropna()
            if len(time_diff) > 0:
                mean_diff = np.mean(time_diff)
                features['sampling_rate'] = 1000 / mean_diff if mean_diff > 0 else 0
                features['duration'] = (timestamp_data.max() - timestamp_data.min()) / 1000
    
    return pd.DataFrame([features])

# ============================================
# 5. Load and Process All Data (Updated)
# ============================================
print("\n" + "=" * 50)
print(" Loading and processing all data...")
print("=" * 50)

all_features = []
all_labels = []
file_info = []
failed_files = []

# Process GOOD files
print(f" Processing {len(good_files)} GOOD files...")
for i, filename in enumerate(good_files):
    try:
        file_path = os.path.join(GOOD_DIR, filename)
        
        # Read file
        df = read_csv_safe(file_path)
        
        if df is None or len(df) == 0:
            print(f"     Skipping {filename}: empty data")
            failed_files.append(filename)
            continue
        
        # Extract features
        features_df = extract_features(df)
        
        if not features_df.empty:
            all_features.append(features_df)
            all_labels.append(1)  # Good = 1
            
            file_info.append({
                'filename': filename,
                'label': 'Good',
                'rows': len(df),
                'features': len(features_df.columns)
            })
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(good_files)} files")
        else:
            print(f"     Skipping {filename}: failed to extract features")
            failed_files.append(filename)
            
    except Exception as e:
        print(f"    Error in {filename}: {e}")
        failed_files.append(filename)

# Process BAD files
print(f"\n Processing {len(bad_files)} BAD files...")
for i, filename in enumerate(bad_files):
    try:
        file_path = os.path.join(BAD_DIR, filename)
        
        # Read file
        df = read_csv_safe(file_path)
        
        if df is None or len(df) == 0:
            print(f"     Skipping {filename}: empty data")
            failed_files.append(filename)
            continue
        
        # Extract features
        features_df = extract_features(df)
        
        if not features_df.empty:
            all_features.append(features_df)
            all_labels.append(0)  # Bad = 0
            
            file_info.append({
                'filename': filename,
                'label': 'Bad',
                'rows': len(df),
                'features': len(features_df.columns)
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(bad_files)} files")
        else:
            print(f"   ⚠️  Skipping {filename}: failed to extract features")
            failed_files.append(filename)
            
    except Exception as e:
        print(f"    Error in {filename}: {e}")
        failed_files.append(filename)

# ============================================
# 6. Data Cleaning and Processing (New)
# ============================================
if not all_features:
    print(" Failed to load any data!")
    exit(1)

# Merge all features
X = pd.concat(all_features, ignore_index=True)
y = np.array(all_labels)

print(f"\n Successfully loaded {len(all_features)} samples!")
print(f" Data distribution: Good={sum(y == 1)}, Bad={sum(y == 0)}")
print(f" Original number of features: {X.shape[1]}")

# Clean NaN and invalid values
print("\n Cleaning data...")

# 1. Replace NaN
X = X.fillna(0)

# 2. Replace inf with defined values
X = X.replace([np.inf, -np.inf], 0)

# 3. Remove columns where all values are zero
zero_cols = X.columns[(X == 0).all()]
if len(zero_cols) > 0:
    print(f"   Removing {len(zero_cols)} columns with all zero values")
    X = X.drop(columns=zero_cols)

# 4. Remove columns with zero variance (non-informative)
low_var_cols = X.columns[X.std() < 0.001]
if len(low_var_cols) > 0:
    print(f"   Removing {len(low_var_cols)} columns with very low variance")
    X = X.drop(columns=low_var_cols)

print(f" Data cleaned!")
print(f" Number of features after cleaning: {X.shape[1]}")

# Display feature information
print(f"\n Examples of extracted features:")
if X.shape[1] > 0:
    feature_samples = list(X.columns)[:min(10, X.shape[1])]
    for i, feat in enumerate(feature_samples):
        print(f"   {i+1}. {feat}: {X[feat].mean():.4f}")
else:
    print("     No features available after cleaning")

# ============================================
# 7. Split Data and Scale
# ============================================
if X.shape[1] == 0:
    print(" Not enough features for training!")
    exit(1)

print("\n" + "=" * 50)
print(" Training RandomForest model...")
print("=" * 50)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Data split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 8. Train RandomForest Model
# ============================================
# Create and train the model
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in a leaf
    random_state=42,
    n_jobs=-1,             # Use all processors
    class_weight='balanced'  # Balance classes
)

print("\n Training model...")
rf_model.fit(X_train_scaled, y_train)
print(" Model trained!")

# ============================================
# 9. Evaluate Model
# ============================================
print("\n" + "=" * 50)
print(" Model Performance Evaluation")
print("=" * 50)

# Predict on test data
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f" Overall accuracy: {accuracy:.2%}")
print(f" F1 score: {f1:.2%}")
print(f" Log Loss: {loss:.4f}")

# Detailed classification report
print("\n Detailed classification report:")
print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))

# Confusion matrix
print(" Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================
# 10. Visualize Results
# ============================================
print("\n" + "=" * 50)
print(" Visualizing results")
print("=" * 50)

try:
    # رسم بياني يوضح Accuracy و F1 و Loss 
    plt.figure(figsize=(6,4)) 
    metrics = ['Accuracy', 'F1 Score', 'Log Loss'] 
    values = [accuracy, f1, loss] 
    colors = ['green', 'blue', 'red'] 
    plt.bar(metrics, values, color=colors) 
    plt.title("Model Performance Metrics") 
    plt.ylabel("Value") 
    plt.ylim(0,1) # لأن accuracy و f1 بين 0 و 1، والـ loss عادة صغيرة plt.show()

    # 1. Confusion matrix
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad', 'Good'], 
                yticklabels=['Bad', 'Good'])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # 2. Confidence distribution in predictions
    plt.subplot(2, 2, 2)
    confidences = np.max(y_pred_proba, axis=1)
    plt.hist(confidences, bins=20, alpha=0.7, color='green')
    plt.xlabel('Confidence Level')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Confidence Distribution')
    
    # 3. Most important features
    plt.subplot(2, 2, 3)
    feature_importance = rf_model.feature_importances_
    if len(feature_importance) > 0:
        important_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        important_features_names = X.columns[important_features_idx]
        important_features_values = feature_importance[important_features_idx]
        
        plt.barh(range(len(important_features_names)), important_features_values)
        plt.yticks(range(len(important_features_names)), important_features_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
    
    # 4. Data distribution
    plt.subplot(2, 2, 4)
    labels = ['Good', 'Bad']
    sizes = [sum(y == 1), sum(y == 0)]
    colors = ['#4CAF50', '#F44336']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Data Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"  Error in visualization: {e}")

# ============================================
# 11. Save Model
# ============================================
print("\n" + "=" * 50)
print(" Saving model and scaler")
print("=" * 50)

# Save model
model_path = os.path.join(MODELS_DIR, 'rf_handwriting_model.pkl')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)

print(f" Model saved to: {model_path}")
print(f" Scaler saved to: {scaler_path}")

# Save additional information
info = {
    'accuracy': accuracy,
    'f1_score': f1,
    'num_samples': len(X),
    'num_features': X.shape[1],
    'feature_names': list(X.columns),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'good_files': len(good_files),
    'bad_files': len(bad_files),
    'failed_files': len(failed_files)
}

info_path = os.path.join(MODELS_DIR, 'model_info.pkl')
joblib.dump(info, info_path)
print(f" Model information saved to: {info_path}")

# Save list of failed files
if failed_files:
    failed_path = os.path.join(MODELS_DIR, 'failed_files.txt')
    with open(failed_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed_files))
    print(f" List of failed files saved to: {failed_path}")

# ============================================
# 12. Final Model Information
# ============================================
print("\n" + "=" * 60)
print(" Model training and evaluation completed successfully!")
print("=" * 60)

print(f"\n Results summary:")
print(f"   Test accuracy: {accuracy:.2%}")
print(f"   F1 score: {f1:.2%}")
print(f"   Total samples: {len(X)}")
print(f"   Number of features: {X.shape[1]}")

print(f"\n Saved files:")
print(f"    Model: models/rf_handwriting_model.pkl")
print(f"    Scaler: models/scaler.pkl")
print(f"    Information: models/model_info.pkl")
print(f"    Visualizations: results/model_results.png")

print(f"\n Data distribution:")
print(f"    GOOD: {len(good_files)} files ({sum(y == 1)} samples)")
print(f"    BAD: {len(bad_files)} files ({sum(y == 0)} samples)")
if failed_files:
    print(f"     Failed: {len(failed_files)} files")

print("\n Next steps:")
print("   1. Use the model to predict quality of new files")
print("   2. Collect more data to improve accuracy")
print("   3. Try other algorithms (XGBoost, Neural Networks)")

print("\n" + "=" * 60)

# Additional report
print("\n Data report:")
print(f"   - Successful files: {len(all_features)}")
print(f"   - Failed files: {len(failed_files)}")
print(f"   - Success rate: {len(all_features)/(len(all_features)+len(failed_files)):.2%}")

if failed_files:
    print("\n Files that failed processing (first 5):")
    for i, f in enumerate(failed_files[:5]):
        print(f"   {i+1}. {f}")
    if len(failed_files) > 5:
        print(f"   ... and {len(failed_files)-5} more files")