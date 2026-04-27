#!/usr/bin/env python3
"""
KNN Risk Model Inference
========================
Dự đoán nhãn rủi ro (low / medium / high) cho một hoặc nhiều nhân viên
dựa trên model KNN đã huấn luyện.

Đầu vào: feature dict / list hoặc DataFrame
Đầu ra: risk_label + risk_score_numeric (low=0.1, medium=0.5, high=1.0)
"""

import argparse
import joblib
import pandas as pd
import numpy as np


def load_model(model_dir: str):
    """Load model KNN + danh sách cột feature."""
    model = joblib.load(f"{model_dir}/knn_risk_model.pkl")
    cols = joblib.load(f"{model_dir}/feature_columns.pkl")
    return model, cols


def predict_risk(model, feature_cols, input_data):
    """
    Dự đoán rủi ro.
    
    Parameters
    ----------
    model : sklearn Pipeline
        Pipeline đã huấn luyện (SMOTE không dùng trong inference, chỉ scaler+KNN)
    feature_cols : list[str]
        Thứ tự cột feature đầu vào
    input_data : dict | list[dict] | pd.DataFrame
        Dữ liệu đầu vào với đủ 4 feature
    
    Returns
    -------
    pd.DataFrame với cột risk_label và risk_score_numeric
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data phải là dict, list[dict], hoặc pd.DataFrame")

    # Kiểm tra columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[feature_cols].values
    labels = model.predict(X)

    label_to_score = {'low': 0.1, 'medium': 0.5, 'high': 1.0}
    df['risk_label'] = labels
    df['risk_score_numeric'] = df['risk_label'].map(label_to_score)

    return df


def main():
    parser = argparse.ArgumentParser(description='KNN Risk Inference')
    parser.add_argument('--model-dir', type=str, default='/mnt/agents/output/models',
                        help='Thư mục chứa model artifacts')
    parser.add_argument('--kpi', type=float, required=True,
                        help='KPI completion rate (0.0 - 1.0)')
    parser.add_argument('--delay', type=float, required=True,
                        help='Check-in delay days')
    parser.add_argument('--sentiment', type=float, required=True,
                        help='Feedback sentiment score (-1.0 đến 1.0)')
    parser.add_argument('--obj-ratio', type=float, required=True,
                        help='Objective participation ratio')
    args = parser.parse_args()

    model, cols = load_model(args.model_dir)
    sample = {
        'kpi_completion_rate': args.kpi,
        'checkin_delay_days': args.delay,
        'feedback_sentiment_score': args.sentiment,
        'objective_participation_ratio': args.obj_ratio,
    }
    result = predict_risk(model, cols, sample)
    print(result[['risk_label', 'risk_score_numeric']].to_string(index=False))


if __name__ == '__main__':
    main()
