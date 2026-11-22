"""
Member 2 - Model Evaluation
Module: member2_recommendation/src/evaluate_model.py
"""

import json
import os
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def evaluate_predictions(
    model: ALSModel,
    eval_df: DataFrame,
    dataset_name: str = "Validation",
    cold_start_strategy: str = "drop",
) -> tuple[dict, DataFrame]:
    """
    Generates rating predictions on eval_df and computes RMSE and MAE.
    
    Returns:
        tuple of (metrics_dict, predictions_df)
    """
    # Ensure cold-start strategy is enforced
    if hasattr(model, "setColdStartStrategy"):
        model.setColdStartStrategy(cold_start_strategy)

    print(f"[evaluate_model] Generating predictions on {dataset_name} dataset...")
    predictions = model.transform(eval_df)

    # Filter out NaNs if drop strategy is requested (defensive check)
    clean_predictions = predictions.filter(
        ~F.isnan(F.col("prediction")) & F.col("prediction").isNotNull()
    )

    pred_count = clean_predictions.count()
    original_count = eval_df.count()

    # Evaluators
    rmse_eval = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    mae_eval = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction"
    )

    rmse = rmse_eval.evaluate(clean_predictions)
    mae = mae_eval.evaluate(clean_predictions)

    metrics = {
        "dataset": dataset_name,
        "original_records": original_count,
        "evaluated_predictions": pred_count,
        "dropped_cold_start_records": original_count - pred_count,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }

    print(f"[evaluate_model] {dataset_name} Results -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']} "
          f"({pred_count:,} evaluated, {metrics['dropped_cold_start_records']} cold-start dropped)")

    return metrics, clean_predictions


def save_evaluation_metrics(metrics: dict, json_path: str, csv_path: str = None) -> None:
    """Exports evaluation metrics to JSON and CSV formats."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluate_model] Saved metrics JSON -> {json_path}")

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        # Flatten dictionary for tabular CSV export
        df = pd.DataFrame([metrics])
        df.to_csv(csv_path, index=False)
        print(f"[evaluate_model] Saved metrics CSV -> {csv_path}")
