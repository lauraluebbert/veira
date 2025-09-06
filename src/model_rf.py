import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.utils import shuffle

from utils import data_to_use

# Set random seeds
SEED = 42
np.random.seed(SEED)
# tf.random.set_seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Load data
data_date = "2025-05-02_1814"

data_path = "XXX"
col_meta_path = "XXX"

# data_df_raw = pd.read_csv(raw_data_path, low_memory=False)
data_df = pd.read_csv(data_path, low_memory=False)
col_meta = pd.read_excel(col_meta_path, sheet_name="columns_metadata")
path_meta = pd.read_excel(col_meta_path, sheet_name="features_for_models")


# Utils


def clean_feature_name(s):
    # Split on first '__'
    if "__" in s:
        prefix, remainder = s.split("__", 1)
    else:
        return s  # No '__', return as-is

    # If the prefix is 'cat', remove everything after the last '_' (only necessary when one-hot encoding cat features)
    if prefix == "cat" and "_" in remainder:
        return remainder.rsplit("_", 1)[0]
    else:
        return remainder

    return remainder

# Plot confusion matrices


def plot_confusion_matrices(y_test_dict, y_pred_prob_dict, roc_curve_values, pathogen):
    """
    Plot confusion matrices and print performance metrics for each model.

    Parameters:
        y_test_dict (dict): Mapping model type → y_test (true labels)
        y_pred_prob_dict (dict): Mapping model type → predicted probabilities
        roc_curve_values (dict): Mapping model type → [fpr, tpr, thresholds, auc]
        pathogen (str): Used in plot titles
    """
    for model_type in y_pred_prob_dict:
        y_test = y_test_dict[model_type]
        y_pred_prob = y_pred_prob_dict[model_type]
        fpr, tpr, thresholds, _ = roc_curve_values[model_type]

        # Get optimal threshold using Youden's J
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Classify predictions
        y_pred = (y_pred_prob >= optimal_threshold).astype(int)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred)

        # Print metrics
        print(f"{pathogen} - {model_type}")
        print(f"  Threshold (based on J score): {optimal_threshold:.2f}")
        print(f"  Sensitivity (Recall):  {sensitivity:.3f}")
        print(f"  Specificity:           {specificity:.3f}")
        print(f"  Accuracy:              {accuracy:.3f}")

        # Plot confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        labels = ['Negative', 'Positive']
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {model_type} ({pathogen})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        save_path = f"figures/model_rf/rf_confusion_matrix_{pathogen}_{model_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

# Plot ROC


def plot_roc(roc_curve_values, models, pathogen, colors=None):
    # Plot the ROC curve using ax.plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fontsize = 12

    for i, model_type in enumerate(models.keys()):
        if colors:
            color = colors[i]
        else:
            color = None
        [fpr, tpr, thresholds, auc_score] = roc_curve_values[model_type]

        # # Mark the point corresponding to threshold = 0.5 (or closest)
        # threshold_index = np.argmin(np.abs(thresholds - 0.5))
        # ax.scatter(fpr[threshold_index], tpr[threshold_index], color=color, marker='+', s=80, lw=0.75)

        # Find optimal threshold where probability to converted to 'positive' using Youden’s J statistic (J = TPR - FPR)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        # Mark optimal threshold point
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color=color,
                   edgecolor='black', marker='+', s=80, lw=0.75)
        # print(f"{model_type}:TPR range: {tpr.min():.3f} tso {tpr.max():.3f}")
        # print(f"{model_type}:FPR range: {fpr.min():.3f} to {fpr.max():.3f}")
        print(
            f"{model_type}: Youden’s J scores range: {j_scores.min():.3f} to {j_scores.max():.3f}")
        print(f'{model_type}: Probability threshold to positive based on Youden’s J scores = {optimal_threshold:.2f}')

        ax.plot(
            fpr, tpr, label=f'{model_type} (AUC = {auc_score:.2f})', color=color)

    # Diagonal line (chance level)
    ax.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=fontsize+2)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize+2)
    # ax.set_title('ROC Curve', fontsize=fontsize+2)
    ax.legend(loc='lower right', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True, color="lightgrey", ls="--", lw=1, alpha=0.5)
    ax.set_axisbelow(True)

    save_path = f"figures/model_rf/rf_roc_curves_{pathogen}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def get_standard_features():
    data_df_models = data_to_use(data_df)

    num_cols = col_meta[col_meta["data_type"] == "numerical"]["column"].values
    num_cols_present = [
        col for col in num_cols if col in data_df_models.columns]
    cat_cols = col_meta[col_meta["data_type"]
                        == "categorical"]["column"].values
    cat_cols_present = [
        col for col in cat_cols if col in data_df_models.columns]

    return num_cols_present, cat_cols_present

# Preprocessing


def preprocess_data(num_cols_present, cat_cols_present):
    # Preprocessing pipeline
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # One-hot encode categorical variables
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    # # Use OrdinalEncoder for categorical variables (keeps all categories but assumes that order matters - RF cannot take this into account anyway)
    # categorical_pipeline = Pipeline([
    #     ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    # ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, num_cols_present),
        ('cat', categorical_pipeline, cat_cols_present)
    ])

    return preprocessor

# Random forest model

def process_dataset(train_test_data_folder, include = False):
    pathogen = "all-viral"
    num_cols_present, cat_cols_present = get_standard_features()

    with open(f"{train_test_data_folder}/X_train_{pathogen}.pkl", "rb") as f:
        X_train_raw = pickle.load(f)
        # Drop record_id column
        if "record_id" in X_train_raw.columns and include is False:
            X_train_raw = X_train_raw.drop(columns=["record_id"])
    with open(f"{train_test_data_folder}/X_test_{pathogen}.pkl", "rb") as f:
        X_test_raw = pickle.load(f)
        if "record_id" in X_test_raw.columns and include is False:
            X_test_raw = X_test_raw.drop(columns=["record_id"])
    with open(f"{train_test_data_folder}/y_train_{pathogen}.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(f"{train_test_data_folder}/y_test_{pathogen}.pkl", "rb") as f:
        y_test = pickle.load(f)

    # Fit preprocessor on training data and process X
    local_preprocessor = preprocess_data(
        num_cols_present, cat_cols_present).fit(X_train_raw)
    X_train = local_preprocessor.transform(X_train_raw)
    X_test = local_preprocessor.transform(X_test_raw)

    return X_train, X_train_raw, X_test, X_test_raw, y_train, y_test


def build_rf_model(
    pathogen,
    train_test_data_folder, 
    num_cols_present,
    cat_cols_present, 
    random_state=SEED
    ):

    # Load data
    # To load the saved splits from disk:
    with open(f"{train_test_data_folder}/X_train_{pathogen}.pkl", "rb") as f:
        X_train_raw = pickle.load(f)
        # Drop record_id column
        if "record_id" in X_train_raw.columns:
            X_train_raw = X_train_raw.drop(columns=["record_id"])
    with open(f"{train_test_data_folder}/X_test_{pathogen}.pkl", "rb") as f:
        X_test_raw = pickle.load(f)
        if "record_id" in X_test_raw.columns:
            X_test_raw = X_test_raw.drop(columns=["record_id"])
    with open(f"{train_test_data_folder}/y_train_{pathogen}.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(f"{train_test_data_folder}/y_test_{pathogen}.pkl", "rb") as f:
        y_test = pickle.load(f)

    # Fit preprocessor on training data and process X
    local_preprocessor = preprocess_data(
        num_cols_present, cat_cols_present).fit(X_train_raw)
    X_train = local_preprocessor.transform(X_train_raw)
    X_test = local_preprocessor.transform(X_test_raw)

    models_rf = {}
    roc_curve_values_rf = {}
    y_pred_prob_dict_rf = {}
    y_test_dict_rf = {}

    for model_type in ['true', 'scrambled']:
        if model_type == 'scrambled':
            y_train_used = shuffle(
                y_train, random_state=random_state).reset_index(drop=True)
            y_test_used = shuffle(
                y_test, random_state=random_state).reset_index(drop=True)
        else:
            y_train_used = y_train.reset_index(drop=True)
            y_test_used = y_test.reset_index(drop=True)

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train_used)

        # Store model
        models_rf[model_type] = model

        # Store y test
        y_test_dict_rf[model_type] = y_test_used

        # Evaluate
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_prob_dict_rf[model_type] = y_pred_prob

        auc_score = roc_auc_score(y_test_used, y_pred_prob)
        print(f"AUC/ROC for {pathogen} ({model_type}): {auc_score:.4f}")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_curve_values_rf[model_type] = [fpr, tpr, thresholds, auc_score]

    return local_preprocessor, models_rf, y_pred_prob_dict_rf, roc_curve_values_rf, y_test_dict_rf


def get_feature_importances(model, local_preprocessor, pathogen):
    # # Transform full feature set with the same preprocessor to get exact feature names
    # X_transformed = local_preprocessor.transform(
    #     data_to_use(data_df[data_df[f"{pathogen}_label"] != 2]))
    feature_names = local_preprocessor.get_feature_names_out()

    # Validate shape match
    if len(feature_names) != len(model.feature_importances_):
        print(f"[WARNING] Feature count mismatch for {pathogen}:")
        print(f" - {len(feature_names)} feature names")
        print(f" - {len(model.feature_importances_)} importances")
        return None

    # Build importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Feature_clean': [clean_feature_name(f) for f in feature_names],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return importance_df


def plot_feature_importances(importance_df, pathogen):
    fig, ax = plt.subplots(figsize=(8, 5))
    fontsize = 12

    # Only plot features with Importance > 0.001
    importance_df_temp = importance_df[importance_df['Importance'] > 0.001]

    ax.barh(importance_df_temp['Feature'], importance_df_temp['Importance'])

    # Annotate top features
    top_features = importance_df[['Feature', 'Importance']].head(15)

    def clean_feature_label(feature):
        if "cat" in feature:
            # Replace the last "_" with "="
            idx = feature.rfind("_")
            if idx != -1:
                feature = feature[:idx] + "=" + feature[idx+1:]
        feature = feature.replace("num__", "num: ").replace("cat__", "cat: ").replace("_crf", "").replace("___", "_")
        return feature

    text_str = "Top features:\n" + "\n".join(
        [f'{clean_feature_label(feature)} ({importance:.4f})' for feature,
         importance in top_features.itertuples(index=False)]
    )
    ax.text(
        0.52, 0.95, text_str, transform=ax.transAxes, fontsize=fontsize,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8,
                  edgecolor='gray', boxstyle='round,pad=0.5')
    )

    ax.set_xlabel('Importance', fontsize=fontsize+2)
    ax.set_ylabel('Feature', fontsize=fontsize+2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.set_title(f'{pathogen}: Feature Importances from Random Forest')
    ax.margins(y=0.01)
    ax.set_yticklabels([])  # Optional: hide y-axis labels

    save_path = f"figures/model_rf/rf_feature_importance_{pathogen}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Create new feature sets


def create_feature_sets(importance_df, pathogen):
    feature_sets = {}

    # Keep all features
    feature_sets["all"] = "all"

    # Use only top 20, 10, 5 features
    top_n = [20, 10, 5]
    for n in top_n:
        feature_sets[f"top_{n}"] = importance_df['Feature'].head(n).tolist()

    # # Test with only numerical features
    # feature_sets["num_only"] = col_meta[col_meta["data_type"] == "numerical"]["column"].values

    try:
        # Test with features expected to be relevant for malaria
        feature_sets["manual"] = list(
            path_meta[path_meta[f"{pathogen}_manual"] == 1]["column"].values)
    except KeyError:
        print(f"No manual feature set available for pathogen {pathogen}.")

    # Test with only with features that are usually always collected in hospitals
    feature_sets["freq_avail"] = list(
        path_meta[path_meta["freq_avail"] == 1]["column"].values)

    return feature_sets

# Rerun rf model


def model_new_feature_sets(
    feature_sets, 
    pathogen, 
    train_test_data_folder, 
    num_cols_present, 
    cat_cols_present, 
    random_state=SEED
    ):

    # New containers to avoid overwriting
    models_subset = {}
    preprocessors_subset = {}
    roc_curve_values_subset = {}
    y_pred_prob_dict_subset = {}
    y_test_dict_subset = {}

    # Loop over each named feature set
    for feature_set_name, grouped_features in feature_sets.items():
        print(f"\nTraining model with feature set: {feature_set_name}")

        data_df_models = data_to_use(data_df)

        if grouped_features == "all":
            # Get list of all features
            selected_features_present = data_df_models.columns

        else:
            # Get list of actual features used in the original DataFrame
            all_available_features = data_df_models.columns

            # Clean up feature names (remove "num__" "cat__")
            grouped_features = [clean_feature_name(
                f) for f in grouped_features]

            # Determine original features that are both in the selected set and in the data
            selected_features_present = [
                col for col in grouped_features if col in all_available_features]

        # Identify their types using col_meta
        num_cols = col_meta[col_meta["data_type"]
                            == "numerical"]["column"].values
        cat_cols = col_meta[col_meta["data_type"]
                            == "categorical"]["column"].values

        num_cols_present = [
            col for col in num_cols if col in selected_features_present]
        cat_cols_present = [
            col for col in cat_cols if col in selected_features_present]

        local_preprocessor, models_rf, y_pred_prob_dict_rf, roc_curve_values_rf, y_test_dict_rf = build_rf_model(
            pathogen, train_test_data_folder, num_cols_present, cat_cols_present, random_state=random_state
        )

        models_subset[feature_set_name] = models_rf
        preprocessors_subset[feature_set_name] = local_preprocessor
        roc_curve_values_subset[feature_set_name] = roc_curve_values_rf
        y_pred_prob_dict_subset[feature_set_name] = y_pred_prob_dict_rf
        y_test_dict_subset[feature_set_name] = y_test_dict_rf

    return models_subset, preprocessors_subset, roc_curve_values_subset, y_pred_prob_dict_subset, y_test_dict_subset


def plot_feature_distributions(features_to_plot, importance_df, pathogen):
    # Set up plot grid
    n_cols = 4
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(
        16, n_rows * 3), sharex=False, sharey=False)
    axes = axes.flatten()

    # Plot histogram for test/train data only
    dataset = "test"
    X_path = "XXX"
    y_path = "XXX"
    # Load X and y
    with open(X_path, "rb") as f:
        data_df = pickle.load(f)
    with open(y_path, "rb") as f:
        y = pickle.load(f)
    data_df[f"{pathogen}_label"] = y

    # Plot histograms
    for i, feature in enumerate(features_to_plot):
        data_df_temp = data_df.copy()

        if feature == "date_crf":
            # Ensure the date column is datetime type for plotting
            data_df_temp[feature] = pd.to_datetime(
                data_df_temp[feature], errors='coerce')

        # Map numeric labels to readable categories
        label_col = f"{pathogen}_label"
        data_df_temp[label_col] = data_df_temp[label_col].map(
            {1: "positive", 0: "negative", 2: "undefined"})

        # Define color palette
        palette = {"positive": "red", "negative": "black", "undefined": "grey"}

        ax = axes[i]
        sns.histplot(
            data=data_df_temp,
            x=feature,
            hue=label_col,
            bins=30,
            element='bars',
            multiple='stack',
            stat='count',  # Use 'proportion' for normalized values
            palette=palette,
            ax=ax,
            lw=0,
            alpha=0.7
        )
        # Remove legend title
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(None)

        if feature == "date_crf":
            # Format x-axis by month
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        imp_value = np.max(
            importance_df[importance_df["Feature_clean"] == feature]['Importance'].values)
        ax.set_title(
            f'{feature}\n(Max importance: {imp_value:.4f})', fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Sample count")
        # # Turn off legend for each subplot
        # if ax.get_legend():
        #     ax.get_legend().remove()

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # plt.suptitle("Feature Distributions by Infection Status",
    #              fontsize=16, y=1.02)

    save_path = f"figures/model_rf/rf_feature_distributions_{pathogen}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
