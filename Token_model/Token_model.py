#%%
# train_pipeline.py
import pandas as pd
import numpy as np
import shap
import joblib
import json
from catboost import CatBoostClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
#%%
# ========== STEP 1: 特徵處理 ==========
def feature_ETL(df, df2):
    df = df.dropna()
    df2 = df2.dropna()
    df2 = df2[['contract','marketCap','price']]
    merged_df = pd.merge(df, df2, on='contract', how='inner')
    merged_df = merged_df[~((merged_df['marketCap_x'] == 0) & (merged_df['marketCap_y'] == 0))]

    # 計算比率
    merged_df['marketCap_ratio'] = (merged_df['marketCap_y'] - merged_df['marketCap_x']) / merged_df['marketCap_x']
    merged_df['marketCap_ratio'] = merged_df['marketCap_ratio'].astype(float)

    # 分類標籤 1: 幅度分類
    conditions = [merged_df['marketCap_ratio'].abs() >= 0.005]
    merged_df['amplitude_class'] = np.select(conditions, [1], default=0)

    # 分類標籤 2: 漲跌分類
    merged_df['direction_class'] = (merged_df['marketCap_ratio'] > 0).astype(int)

    merged_df = merged_df.drop(columns=['price_y','marketCap_y','marketCap_ratio'])
    return merged_df

# ========== STEP 2: 載入與合併資料 ==========
def load_all_data(df_list, df2_list):
    all_merged = []
    for i, file in enumerate(df_list):
        df = pd.read_csv(f'/Users/jeremy/Desktop/code_reset/token_features_with_risk_{file}.csv')
        df2 = pd.read_csv(f'/Users/jeremy/Desktop/code_reset/token_features_with_risk_{df2_list[i]}.csv')
        all_merged.append(feature_ETL(df, df2))
    return pd.concat(all_merged, ignore_index=True)

def load_all_data2(df_list, df2_list):
    all_merged = []
    for i, file in enumerate(df_list):
        df = pd.read_csv(f'/Users/jeremy/Desktop/code_reset/Base_token_features_with_risk_{file}.csv')
        df2 = pd.read_csv(f'/Users/jeremy/Desktop/code_reset/Base_token_features_with_risk_{df2_list[i]}.csv')
        all_merged.append(feature_ETL(df, df2))
    return pd.concat(all_merged, ignore_index=True)
# ========== STEP 3: 分類平衡 ==========
def balance_classes(df, label_col):
    classes = df[label_col].unique()
    min_count = min(df[label_col].value_counts())
    balanced = []
    for c in classes:
        df_c = df[df[label_col] == c]
        df_c_bal = resample(df_c, replace=(len(df_c) < min_count), n_samples=min_count, random_state=42)
        balanced.append(df_c_bal)
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

# ========== STEP 4: 模型訓練與儲存 ==========
def train_and_save_model_1(balanced_df, label_col, model_name):
    X = balanced_df.drop(['contract', 'amplitude_class', 'direction_class'], axis=1)
    y = balanced_df[label_col]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    param_grid = {
    'iterations': [50,100,300, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'l2_leaf_reg': [3, 5, 7]}

    catboost_model = CatBoostClassifier(random_seed=42, verbose=100)

    grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_model = grid_search.best_estimator_


    class_weights = {0: 1, 1: 1} 

    best_model_with_weights = CatBoostClassifier(
        iterations=best_model.get_params()['iterations'],
        depth=best_model.get_params()['depth'],
        learning_rate=best_model.get_params()['learning_rate'],
        l2_leaf_reg=best_model.get_params()['l2_leaf_reg'],
        class_weights=class_weights,  # 添加類別權重
        random_seed=42,
        verbose=100)

    model = best_model_with_weights
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    # SHAP 特徵重要性
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, show=False, plot_type="bar")
    plt.title(f'{model_name} Feature Importance Bar')
    plt.tight_layout()
    plt.savefig(f'{model_name}_shap_bar.png')
    plt.close()

    shap.summary_plot(shap_values, X_train)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{model_name}_shap_Feature.png')
    plt.close()

    # 儲存模型與特徵欄位
    joblib.dump(model, f'{model_name}.pkl')
    with open(f'{model_name}_features.json', 'w') as f:
        json.dump(feature_names, f)

def train_and_save_model_2(balanced_df, label_col, model_name):
    X = balanced_df.drop(['contract', 'amplitude_class', 'direction_class','marketCap_x'], axis=1)
    y = balanced_df[label_col]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    param_dist = {
    'iterations': randint(100, 1000),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.3),
    'l2_leaf_reg': randint(1, 10)}

    random_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(random_seed=42, verbose=0),
        param_distributions=param_dist,
        n_iter=50,
        scoring='accuracy',
        cv=3,
        random_state=42)

    random_search.fit(X_train, y_train)
    print("Best parameters found: ", random_search.best_params_)

    best_model = random_search.best_estimator_

    class_weights = {0: 1, 1: 1} 

    best_model_with_weights = CatBoostClassifier(
        iterations=best_model.get_params()['iterations'],
        depth=best_model.get_params()['depth'],
        learning_rate=best_model.get_params()['learning_rate'],
        l2_leaf_reg=best_model.get_params()['l2_leaf_reg'],
        class_weights=class_weights,  # 添加類別權重
        random_seed=42,
        verbose=100)

    model = best_model_with_weights
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    # SHAP 特徵重要性
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, show=False, plot_type="bar")
    plt.title(f'{model_name} Feature Importance Bar')
    plt.tight_layout()
    plt.savefig(f'{model_name}_shap_bar.png')
    plt.close()

    shap.summary_plot(shap_values, X_train)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{model_name}_shap_Feature.png')
    plt.close()

    # 儲存模型與特徵欄位
    joblib.dump(model, f'{model_name}.pkl')
    with open(f'{model_name}_features.json', 'w') as f:
        json.dump(feature_names, f)
#%%
# ========== 主程式 ==========

df_list = ['0940','1040','1140','1240']
df2_list = ['1040','1140','1240','1340']

merged_df = load_all_data(df_list, df2_list)

# 訓練振幅模型
amplitude_df = balance_classes(merged_df, 'amplitude_class')
train_and_save_model_1(amplitude_df, 'amplitude_class', 'ethereum_amplitude_model')

# 訓練漲跌模型
direction_df = balance_classes(merged_df, 'direction_class')
train_and_save_model_2(direction_df, 'direction_class', 'ethereum_direction_model')


df_list = ['1340','1440','1540','1640']
df2_list = ['1440','1540','1640','1740']

merged_df = load_all_data2(df_list, df2_list)

# 訓練振幅模型
amplitude_df = balance_classes(merged_df, 'amplitude_class')
train_and_save_model_1(amplitude_df, 'amplitude_class', 'base_amplitude_model')

# 訓練漲跌模型
direction_df = balance_classes(merged_df, 'direction_class')
train_and_save_model_2(direction_df, 'direction_class', 'base_direction_model')

print("\n✅ 所有模型訓練完成，已儲存至本地。")

# %%
