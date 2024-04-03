import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import plot_tree
import shap
import lime
import lime.lime_tabular

file_paths = {
    'X_train': 'preprocessed_data/X_train.pkl',
    'X_test': 'preprocessed_data/X_test.pkl',
    'y_train': 'preprocessed_data/y_train.pkl',
    'y_test': 'preprocessed_data/y_test.pkl'
}

if not os.path.exists('preprocessed_data'):
    os.makedirs('preprocessed_data')

def load_if_exists(file_paths):
    if all(os.path.exists(path) for path in file_paths.values()):
        print("Loading preprocessed datasets...")
        return (
            pd.read_pickle(file_paths['X_train']),
            pd.read_pickle(file_paths['X_test']),
            pd.read_pickle(file_paths['y_train']),
            pd.read_pickle(file_paths['y_test'])
        )
    else:
        return None, None, None, None

X_train, X_test, y_train, y_test = load_if_exists(file_paths)
# If datasets don't exist, proceed with splitting and saving
if X_train is None or X_test is None or y_train is None or y_test is None:
    print("Loading dataset...")
    data = pd.read_csv('train/train.csv')
    print("Dataset loaded successfully.\n")

    # Initial Data Check
    print(f"Initial dataset shape: {data.shape}")
    print("Checking for null values...\n")
    print(data.isnull().sum())

    columns_with_nulls = data.columns[data.isnull().any()]
    null_percentage = data[columns_with_nulls].isnull().mean() * 100
    print(f"Columns with nulls and their percentage: \n{null_percentage}\n")
    # Drop columns with significant missing values if necessary, e.g., threshold > 50%
    threshold = 50
    columns_to_drop = null_percentage[null_percentage > threshold].index
    data.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns with more than {threshold}% missing values. Current shape: {data.shape}\n")
    print("Handling missing values...")
    data.fillna('unknown', inplace=True)

    # Check if 'unknown' value is present in any column
    unknown_values_present = data.isin(['unknown']).any().any()
    if unknown_values_present:
        print("Unknown values are present in the DataFrame.")
    else:
        print("No unknown values found in the DataFrame.")

    # Unique Values Check for Categorical Columns
    print("Unique values in each categorical column:\n")
    categorical_features = ['platform', 'city', 'device', 'reference']
    for feature in categorical_features:
        unique_values = data[feature].nunique()
        print(f"{feature}: {unique_values} unique values")
    data['is_clickout'] = (data['action_type'] == 'clickout item').astype(int)
    data['target'] = data.groupby('session_id')['is_clickout'].transform('last')

    # Create new Features
    print("Engineering New features...")
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data.sort_values(by=['session_id', 'timestamp'], inplace=True)
    data['session_duration'] = (data.groupby('session_id')['timestamp'].transform('max') - 
                                data.groupby('session_id')['timestamp'].transform('min')).dt.total_seconds()
    data['actions_before_last'] = data.groupby('session_id').cumcount()

    # Encoding Categorical Variables
    print("Encoding categorical variables...")
    # Target Encoding for 'city' due to high cardinality
    te = TargetEncoder()
    data['city_encoded'] = te.fit_transform(data['city'], data['target'])

    # Label Encoding for 'reference'
    label_encoder = LabelEncoder()
    data['reference_encoded'] = label_encoder.fit_transform(data['reference'])

    # Dropping original columns to avoid duplicating information
    data.drop(['city', 'reference'], axis=1, inplace=True)

    # OneHot Encoding for remaining categorical variables
    data = pd.get_dummies(data, columns=['platform', 'device'], drop_first=True)

    # Preparing Data for Modeling
    features_to_drop = ['user_id', 'session_id', 'timestamp', 'action_type', 'is_clickout']
    X = data.drop(features_to_drop + ['target'], axis=1)
    y = data['target']

    unknown_columns = data.columns[data.isin(['unknown']).any()]
    if len(unknown_columns) > 0:
        print("Unknown values are present in the following columns after preprocessing:")
        print(unknown_columns)
    else:
        print("No unknown values found in the DataFrame after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split into training and test sets. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}\n")

    X_train.to_pickle(file_paths['X_train'])
    X_test.to_pickle(file_paths['X_test'])
    y_train.to_pickle(file_paths['y_train'])
    y_test.to_pickle(file_paths['y_test'])
    print("Datasets saved to preprocessed_data folder.")
else:
    print("Datasets loaded successfully.")


# Define file paths for the selected features
feature_dir = 'featureReduction'
if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)
selected_features_path = os.path.join(feature_dir, 'selected_features.pkl')

def load_selected_features(selected_features_path):
    if os.path.exists(selected_features_path):
        return pd.read_pickle(selected_features_path)
    return None

selected_features = load_selected_features(selected_features_path)

if selected_features is None:

    # Sample a portion of X_train and y_train for MI analysis to save computation time
    sample_size = len(X_train) // 1000
    X_train_sample = X_train.sample(n=sample_size, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]

    # Mutual Information Analysis on the sampled data
    print("Performing Mutual Information analysis on sampled data...")
    mi_scores = mutual_info_classif(X_train_sample, y_train_sample, random_state=42)
    mi_scores_series = pd.Series(mi_scores, index=X_train_sample.columns).sort_values(ascending=False)
    mi_scores_df = pd.DataFrame(mi_scores_series, columns=['Mutual Information'])
    print("Feature Importance based on Mutual Information (Sampled Data):")
    print(mi_scores_df.head())
    plt.figure(figsize=(10, 6))
    mi_scores_series.plot(kind='bar')
    plt.title('Feature Importance based on Mutual Information (Sampled Data)')
    plt.ylabel('Mutual Information Score')
    plt.show()
    mi_threshold = np.percentile(mi_scores_series, 75)
    selected_features = mi_scores_series[mi_scores_series > mi_threshold].index
    selected_features.to_series().to_pickle(selected_features_path)
    print(f"Selected features saved to {selected_features_path}")
else:
    print(f"Selected features loaded to successfully.")

print(f"Number of selected features based on MI scores above the 75th percentile: {len(selected_features)}")
print(f"Selected features: {list(selected_features)}")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Select the first 10,000 rows from X_train. Delete it if you want to use the full dataset.
sample_size = min(10000, len(X_train))

X_train_selected = X_train_selected.sample(n=sample_size, random_state=42)
y_train = y_train.loc[X_train_selected.index]
X_test_selected = X_test_selected.sample(n=sample_size, random_state=42)
y_test = y_test.loc[X_test_selected.index]

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

classification_folder = "classification"
if not os.path.exists(classification_folder):
    os.makedirs(classification_folder)

classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

evaluation_metrics = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    
    # Scale features for classifiers that benefit from it
    if name in ["K-Nearest Neighbors", "Neural Network"]:
        X_train_scaled, X_test_scaled = scale_features(X_train_selected, X_test_selected)
        clf.fit(X_train_scaled, y_train)
        predictions = clf.predict(X_test_scaled)
        proba_predictions = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
    else:
        clf.fit(X_train_selected, y_train)
        predictions = clf.predict(X_test_selected)
        proba_predictions = clf.predict_proba(X_test_selected)[:, 1] if hasattr(clf, "predict_proba") else None
    
    if name in ["Random Forest"]:
        tree_index = 0
        selected_tree = clf.estimators_[tree_index]

        plt.figure(figsize=(20, 10))
        plot_tree(selected_tree, filled=True, fontsize=10, feature_names=X_train_selected.columns.tolist(), class_names=[str(cls) for cls in y_train.unique()])
        plt.title(f'Decision Tree {tree_index + 1}', fontsize=20)
        plt.savefig(os.path.join(classification_folder, f"decision_tree_{tree_index + 1}.png"))  # Save the decision tree plot
        plt.show()

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    roc_auc = roc_auc_score(y_test, proba_predictions) if proba_predictions is not None else "N/A"
    evaluation_metrics[name] = {"F1 Score": f1, "Precision": precision, "Recall": recall, "ROC AUC": roc_auc}
    print(f"{name} Classification Report:\n{classification_report(y_test, predictions)}")
    print(f"ROC AUC Score: {roc_auc}\n")

plt.figure(figsize=(10, 8))
for name, clf in classifiers.items():
    if "ROC AUC" in evaluation_metrics[name] and evaluation_metrics[name]["ROC AUC"] != "N/A":
        X_test_to_use = X_test_scaled if name in ["K-Nearest Neighbors", "Neural Network"] else X_test_selected
        fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_to_use)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {evaluation_metrics[name]["ROC AUC"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(classification_folder, "ROC_curve.png"))  # Save the ROC curve plot
plt.show()

### Explainable AI (XAI) ###

shap_dir = "shap"
lime_dir = "lime"

if not os.path.exists(shap_dir):
    os.makedirs(shap_dir)
if not os.path.exists(lime_dir):
    os.makedirs(lime_dir)

### Explainable AI (XAI) with RandomForestClassifier ###

print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train_selected, y_train)

print("Initializing SHAP TreeExplainer...")
explainer_shap = shap.TreeExplainer(clf)
print("Sampling a subset for SHAP values calculation...")
X_train_sampled_shap = X_train_selected.sample(100, random_state=42)
print("Calculating SHAP values...")
shap_values = explainer_shap.shap_values(X_train_sampled_shap)
print("Visualizing global feature importance with SHAP...")
shap_fig = shap.summary_plot(shap_values, X_train_sampled_shap, plot_type="bar", show=False)
plt.savefig(os.path.join(shap_dir, "rf_shap_summary_plot.png"))
plt.close()

# LIME Explanation
print("Initializing LIME TabularExplainer...")
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_selected),
    feature_names=X_train_selected.columns.tolist(),
    class_names=['Negative', 'Positive'],
    mode='classification'
)

instance_idx = 0
instance = X_test_selected.iloc[instance_idx].to_numpy().reshape(1, -1)
print(f"Explaining instance {instance_idx} prediction with LIME...")
lime_exp = explainer_lime.explain_instance(
    data_row=instance[0],
    predict_fn=clf.predict_proba,
    num_features=len(X_train_selected.columns)
)
print("Visualizing LIME explanation for the selected instance...")
lime_fig = lime_exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig(os.path.join(lime_dir, "rf_lime_explanation.png"))
plt.close()

print("Completed Explainable AI (XAI) Analysis for RandomForestClassifier.")

### Explainable AI (XAI) with MLPClassifier ###

mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

print("Training the Neural Network...")
mlp_clf.fit(X_train_selected, y_train)

print("Initializing SHAP KernelExplainer for MLP...")
explainer_shap_mlp = shap.KernelExplainer(mlp_clf.predict_proba, shap.sample(X_train_selected, 100))
shap_values_mlp = explainer_shap_mlp.shap_values(shap.sample(X_test_selected, 10))

print("Visualizing SHAP summary plot for MLP...")
shap_fig_mlp = shap.summary_plot(shap_values_mlp, shap.sample(X_test_selected, 10), show=False)
plt.savefig(os.path.join(shap_dir, "mlp_shap_summary_plot.png"))
plt.close()

print("Initializing LIME TabularExplainer for MLP...")
explainer_lime_mlp = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_selected), 
    feature_names=X_train_selected.columns.tolist(),
    class_names=['Negative', 'Positive'],
    mode='classification'
)

print(f"Explaining instance {instance_idx} prediction with LIME for MLP...")
instance = X_test_selected.iloc[instance_idx].values  # This will be a 1D array

lime_exp_mlp = explainer_lime_mlp.explain_instance(
    instance, 
    mlp_clf.predict_proba,
    num_features=len(X_train_selected.columns)
)
print("Visualizing LIME explanation for the selected instance with MLP...")
lime_fig_mlp = lime_exp_mlp.as_pyplot_figure()
plt.tight_layout()
plt.savefig(os.path.join(lime_dir, "mlp_lime_explanation.png"))
plt.close()

print("Completed Explainable AI (XAI) Analysis for MLPClassifier.")
