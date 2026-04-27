#!/usr/bin/env python3
"""
KNN Risk Model Training Pipeline
================================
Huấn luyện model KNN để phân loại rủi ro nhân viên (low / medium / high)
dựa trên đặc trưng KPI, Check-in, Feedback và Objective Participation.

Đầu vào: kpi_okr_data_fixed.xlsx (theo Prisma schema)
Đầu ra: knn_risk_model.pkl + feature_columns.pkl + evaluation report

Thiết kế tham khảo: AI-Feature-System-Design.md
"""

import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def load_raw_data(excel_path: str) -> dict:
    """Đọc toàn bộ các sheet cần thiết từ file Excel."""
    required_sheets = [
        'Users', 'KPI_Assignments', 'Check_Ins',
        'Feedbacks', 'Objectives'
    ]
    xls = pd.ExcelFile(excel_path)
    missing = [s for s in required_sheets if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Missing sheets in Excel: {missing}")

    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in required_sheets}


def build_kpi_completion_rate(kpi_assign: pd.DataFrame) -> pd.DataFrame:
    """Feature 1: Trung bình progress_percentage của KPI_Assignments theo owner."""
    df = kpi_assign.groupby('owner_id')['progress_percentage'].mean().reset_index()
    df.rename(columns={'progress_percentage': 'kpi_completion_rate', 'owner_id': 'user_id'}, inplace=True)
    df['kpi_completion_rate'] = df['kpi_completion_rate'] / 100.0
    return df


def build_checkin_features(check_ins: pd.DataFrame) -> pd.DataFrame:
    """Feature 2: Check-in delay days = khoảng cách trung bình giữa các lần check-in."""
    check_ins['date'] = pd.to_datetime(check_ins['created_at']).dt.date
    stats = []
    for user_id, group in check_ins.groupby('user_id'):
        dates = sorted(group['date'].unique())
        if len(dates) > 1:
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            mean_gap = float(np.mean(gaps))
        else:
            mean_gap = 30.0  # fallback nếu chỉ 1 lần check-in
        stats.append({'user_id': user_id, 'checkin_delay_days': mean_gap})
    return pd.DataFrame(stats)


def build_feedback_sentiment(feedbacks: pd.DataFrame) -> pd.DataFrame:
    """Feature 3: Trung bình sentiment score theo user."""
    mapping = {
        'POSITIVE': 1.0,
        'NEUTRAL': 0.0,
        'MIXED': 0.0,
        'UNKNOWN': 0.0,
        'NEGATIVE': -1.0,
    }
    feedbacks = feedbacks.copy()
    feedbacks['sentiment_score'] = feedbacks['sentiment'].map(mapping)
    df = feedbacks.groupby('user_id')['sentiment_score'].mean().reset_index()
    df.rename(columns={'sentiment_score': 'feedback_sentiment_score'}, inplace=True)
    return df


def build_objective_participation(objectives: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Feature 4: Số Objectives user tham gia / trung bình phòng ban."""
    obj_by_owner = objectives.groupby('owner_id').size().reset_index(name='obj_count')
    obj_by_unit = objectives.groupby('unit_id').size().reset_index(name='unit_avg_obj')

    user_unit = users[['id', 'unit_id']].merge(obj_by_unit, on='unit_id', how='left')
    user_unit = user_unit.merge(obj_by_owner, left_on='id', right_on='owner_id', how='left')
    user_unit['obj_count'] = user_unit['obj_count'].fillna(0)
    user_unit['unit_avg_obj'] = user_unit['unit_avg_obj'].fillna(1.0)
    user_unit['objective_participation_ratio'] = user_unit['obj_count'] / user_unit['unit_avg_obj']

    df = user_unit[['id', 'objective_participation_ratio']].copy()
    df.rename(columns={'id': 'user_id'}, inplace=True)
    return df


def assign_risk_labels(features: pd.DataFrame) -> pd.DataFrame:
    """
    Gán nhãn risk dựa trên rule-based anomaly scoring.
    Logic tham khảo từ AI-Feature-System-Design.md (phần 4.1 + 4.2).
    """
    emp = features[features['role'] == 'EMPLOYEE'].copy()

    baselines = {
        'kpi': (emp['kpi_completion_rate'].mean(), emp['kpi_completion_rate'].std()),
        'delay': (emp['checkin_delay_days'].mean(), emp['checkin_delay_days'].std()),
        'sent': (emp['feedback_sentiment_score'].mean(), emp['feedback_sentiment_score'].std()),
        'obj': (emp['objective_participation_ratio'].mean(), emp['objective_participation_ratio'].std()),
    }

    def score(row):
        s = 0.0
        # KPI thấp -> risk
        if row['kpi_completion_rate'] < baselines['kpi'][0] - 2 * baselines['kpi'][1]:
            s += 0.30
        elif row['kpi_completion_rate'] < baselines['kpi'][0] - baselines['kpi'][1]:
            s += 0.18
        elif row['kpi_completion_rate'] < baselines['kpi'][0] - 0.5 * baselines['kpi'][1]:
            s += 0.08

        # Delay cao -> risk
        if row['checkin_delay_days'] > baselines['delay'][0] + 2 * baselines['delay'][1]:
            s += 0.25
        elif row['checkin_delay_days'] > baselines['delay'][0] + baselines['delay'][1]:
            s += 0.15
        elif row['checkin_delay_days'] > baselines['delay'][0] + 0.5 * baselines['delay'][1]:
            s += 0.08

        # Sentiment thấp -> risk
        if row['feedback_sentiment_score'] < baselines['sent'][0] - 2 * baselines['sent'][1]:
            s += 0.25
        elif row['feedback_sentiment_score'] < baselines['sent'][0] - baselines['sent'][1]:
            s += 0.12

        # Objective ratio thấp -> risk
        if row['objective_participation_ratio'] < baselines['obj'][0] - 2 * baselines['obj'][1]:
            s += 0.20
        elif row['objective_participation_ratio'] < baselines['obj'][0] - baselines['obj'][1]:
            s += 0.12
        return s

    features['risk_score_rule'] = features.apply(score, axis=1)

    def label(score):
        if score >= 0.40:
            return 'high'
        elif score >= 0.18:
            return 'medium'
        return 'low'

    features['risk_label'] = features['risk_score_rule'].apply(label)
    return features


def build_feature_matrix(data: dict) -> pd.DataFrame:
    """Tổng hợp toàn bộ feature engineering."""
    users = data['Users']

    f_kpi = build_kpi_completion_rate(data['KPI_Assignments'])
    f_checkin = build_checkin_features(data['Check_Ins'])
    f_sentiment = build_feedback_sentiment(data['Feedbacks'])
    f_obj = build_objective_participation(data['Objectives'], users)

    features = users[['id', 'company_id', 'full_name', 'unit_id', 'role']].copy()
    features.rename(columns={'id': 'user_id'}, inplace=True)

    features = features.merge(f_kpi, on='user_id', how='left')
    features = features.merge(f_checkin, on='user_id', how='left')
    features = features.merge(f_sentiment, on='user_id', how='left')
    features = features.merge(f_obj, on='user_id', how='left')

    # Fill missing bằng median / 0
    features['kpi_completion_rate'] = features['kpi_completion_rate'].fillna(
        features['kpi_completion_rate'].median())
    features['checkin_delay_days'] = features['checkin_delay_days'].fillna(
        features['checkin_delay_days'].median())
    features['feedback_sentiment_score'] = features['feedback_sentiment_score'].fillna(0.0)
    features['objective_participation_ratio'] = features['objective_participation_ratio'].fillna(
        features['objective_participation_ratio'].median())

    return assign_risk_labels(features)


def train_knn_model(features: pd.DataFrame, output_dir: str, test_size: float = 0.25):
    """
    Huấn luyện KNN classifier với SMOTE + StandardScaler + GridSearchCV.
    Chỉ sử dụng EMPLOYEE (loại trừ ADMIN).
    """
    emp_df = features[features['role'] == 'EMPLOYEE'].copy()
    feature_cols = [
        'kpi_completion_rate',
        'checkin_delay_days',
        'feedback_sentiment_score',
        'objective_participation_ratio',
    ]
    X = emp_df[feature_cols].values
    y = emp_df['risk_label'].values

    print(f"[INFO] Dataset shape: {X.shape}")
    print(f"[INFO] Class distribution: {pd.Series(y).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Pipeline: SMOTE -> Scale -> KNN
    # k_neighbors=1 cho SMOTE để tránh lỗi khi class minor ít mẫu trong CV fold
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=1)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier()),
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan'],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring='f1_macro',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\n[BEST PARAMS] {grid.best_params_}")
    print(f"[BEST CV F1-macro] {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])

    print(f"\n[CLASSIFICATION REPORT]\n{report}")
    print(f"[CONFUSION MATRIX]\n{cm}")

    # Lưu artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'knn_risk_model.pkl')
    cols_path = os.path.join(output_dir, 'feature_columns.pkl')
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    joblib.dump(grid.best_estimator_, model_path)
    joblib.dump(feature_cols, cols_path)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("KNN Risk Model Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best Params: {grid.best_params_}\n")
        f.write(f"Best CV F1-macro: {grid.best_score_:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (low, medium, high):\n")
        f.write(str(cm) + "\n")

    print(f"\n[SAVED] Model -> {model_path}")
    print(f"[SAVED] Columns -> {cols_path}")
    print(f"[SAVED] Report -> {report_path}")

    return grid.best_estimator_, feature_cols


def main():
    parser = argparse.ArgumentParser(description='Train KNN Risk Model from KPI/OKR Excel')
    parser.add_argument('--input', type=str, default='/mnt/agents/upload/kpi_okr_data_fixed.xlsx',
                        help='Path to input Excel file')
    parser.add_argument('--output', type=str, default='/mnt/agents/output/models',
                        help='Directory to save model artifacts')
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Test split ratio (default 0.25)')
    args = parser.parse_args()

    print("[STEP 1] Loading raw data...")
    data = load_raw_data(args.input)

    print("[STEP 2] Building feature matrix...")
    features = build_feature_matrix(data)
    print(f"[INFO] Label distribution:\n{features['risk_label'].value_counts()}")

    print("[STEP 3] Training KNN model...")
    train_knn_model(features, args.output, args.test_size)

    print("\n[DONE] Training pipeline completed successfully.")


if __name__ == '__main__':
    main()
