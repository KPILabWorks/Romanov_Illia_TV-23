import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from faker import Faker
import re
from typing import Dict, List, Tuple, Any
import os

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
fake = Faker(["uk_UA", "en_US"])


class EnergyCompanyDataGenerator:

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        Faker.seed(seed)
        self.companies = [
            "ЕнергоУкраїна",
            "PowerTech",
            "GreenEnergy",
            "UkrElectro",
            "EcoEnergy",
            "MegaWatt",
            "PowerGrid",
            "SolarTech",
        ]
        self.transaction_types = [
            "energy_sale",
            "equipment_purchase",
            "maintenance",
            "infrastructure",
            "renewable_investment",
            "grid_upgrade",
        ]

    def generate_time_series(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        freq: str = "H",
    ) -> pd.DataFrame:

        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_points = len(date_range)

        trend = np.linspace(1000, 1500, n_points)
        seasonal_daily = 200 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        seasonal_weekly = 150 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
        seasonal_monthly = 300 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 30))

        noise = np.random.normal(0, 50, n_points)

        base_values = (
            trend + seasonal_daily + seasonal_weekly + seasonal_monthly + noise
        )

        anomaly_indices = np.random.choice(
            n_points, size=int(0.05 * n_points), replace=False
        )
        anomaly_labels = np.zeros(n_points)
        anomaly_labels[anomaly_indices] = 1

        anomaly_values = base_values.copy()
        for idx in anomaly_indices:

            if np.random.random() > 0.5:
                anomaly_values[idx] += np.random.uniform(500, 1500)
            else:
                anomaly_values[idx] -= np.random.uniform(300, 800)

        data = []
        for i, timestamp in enumerate(date_range):
            transaction = {
                "timestamp": timestamp,
                "company": np.random.choice(self.companies),
                "transaction_type": np.random.choice(self.transaction_types),
                "amount_uah": max(0, anomaly_values[i]),
                "base_amount": max(0, base_values[i]),
                "is_anomaly": anomaly_labels[i],
                "hour": timestamp.hour,
                "day_of_week": timestamp.dayofweek,
                "month": timestamp.month,
                "quarter": timestamp.quarter,
            }
            data.append(transaction)

        df = pd.DataFrame(data)

        df["rolling_mean_24h"] = df["amount_uah"].rolling(window=24, center=True).mean()
        df["rolling_std_24h"] = df["amount_uah"].rolling(window=24, center=True).std()
        df["z_score"] = np.abs(
            (df["amount_uah"] - df["rolling_mean_24h"]) / df["rolling_std_24h"]
        )

        return df


class DataProcessor:

    @staticmethod
    def save_to_json(df: pd.DataFrame, filename: str, nested: bool = True):

        if nested:

            nested_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_records": len(df),
                    "date_range": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat(),
                    },
                    "companies": df["company"].unique().tolist(),
                    "transaction_types": df["transaction_type"].unique().tolist(),
                },
                "statistics": {
                    "total_amount": float(df["amount_uah"].sum()),
                    "average_amount": float(df["amount_uah"].mean()),
                    "anomaly_rate": float(df["is_anomaly"].mean()),
                    "by_company": {},
                },
                "transactions": [],
            }

            for company in df["company"].unique():
                company_data = df[df["company"] == company]
                nested_data["statistics"]["by_company"][company] = {
                    "total_transactions": len(company_data),
                    "total_amount": float(company_data["amount_uah"].sum()),
                    "anomaly_count": int(company_data["is_anomaly"].sum()),
                }

            for _, row in df.iterrows():
                transaction = {
                    "id": int(row.name),
                    "timestamp": row["timestamp"].isoformat(),
                    "company_info": {
                        "name": row["company"],
                        "transaction_type": row["transaction_type"],
                    },
                    "financial_data": {
                        "amount_uah": float(row["amount_uah"]),
                        "base_amount": float(row["base_amount"]),
                        "z_score": (
                            float(row["z_score"])
                            if not pd.isna(row["z_score"])
                            else None
                        ),
                    },
                    "temporal_features": {
                        "hour": int(row["hour"]),
                        "day_of_week": int(row["day_of_week"]),
                        "month": int(row["month"]),
                        "quarter": int(row["quarter"]),
                    },
                    "is_anomaly": bool(row["is_anomaly"]),
                }
                nested_data["transactions"].append(transaction)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(nested_data, f, ensure_ascii=False, indent=2)
        else:

            df_json = df.copy()
            df_json["timestamp"] = df_json["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_json.to_json(filename, orient="records", force_ascii=False, indent=2)

    @staticmethod
    def save_to_xml(df: pd.DataFrame, filename: str):

        root = ET.Element("energy_transactions")

        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "generated_at").text = datetime.now().isoformat()
        ET.SubElement(metadata, "total_records").text = str(len(df))
        ET.SubElement(metadata, "anomaly_rate").text = str(df["is_anomaly"].mean())

        companies = ET.SubElement(metadata, "companies")
        for company in df["company"].unique():
            company_elem = ET.SubElement(companies, "company")
            company_elem.set("name", company)
            company_data = df[df["company"] == company]
            company_elem.set("transactions", str(len(company_data)))
            company_elem.set("total_amount", str(company_data["amount_uah"].sum()))

        transactions = ET.SubElement(root, "transactions")
        for _, row in df.iterrows():
            transaction = ET.SubElement(transactions, "transaction")
            transaction.set("id", str(row.name))
            transaction.set("is_anomaly", str(row["is_anomaly"]))

            ET.SubElement(transaction, "timestamp").text = row["timestamp"].isoformat()
            ET.SubElement(transaction, "company").text = row["company"]
            ET.SubElement(transaction, "type").text = row["transaction_type"]
            ET.SubElement(transaction, "amount").text = str(row["amount_uah"])
            ET.SubElement(transaction, "hour").text = str(row["hour"])
            ET.SubElement(transaction, "day_of_week").text = str(row["day_of_week"])

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def load_and_parse_json(filename: str) -> pd.DataFrame:

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        transactions = []
        for transaction in data["transactions"]:
            flat_transaction = {
                "id": transaction["id"],
                "timestamp": pd.to_datetime(transaction["timestamp"]),
                "company": transaction["company_info"]["name"],
                "transaction_type": transaction["company_info"]["transaction_type"],
                "amount_uah": transaction["financial_data"]["amount_uah"],
                "base_amount": transaction["financial_data"]["base_amount"],
                "z_score": transaction["financial_data"]["z_score"],
                "hour": transaction["temporal_features"]["hour"],
                "day_of_week": transaction["temporal_features"]["day_of_week"],
                "month": transaction["temporal_features"]["month"],
                "quarter": transaction["temporal_features"]["quarter"],
                "is_anomaly": transaction["is_anomaly"],
            }
            transactions.append(flat_transaction)

        df = pd.DataFrame(transactions)
        return df

    @staticmethod
    def parse_xml_with_regex(filename: str) -> Dict:

        with open(filename, "r", encoding="utf-8") as f:
            xml_content = f.read()

        metadata = {}
        metadata["generated_at"] = re.search(
            r"<generated_at>(.*?)</generated_at>", xml_content
        ).group(1)
        metadata["total_records"] = int(
            re.search(r"<total_records>(.*?)</total_records>", xml_content).group(1)
        )
        metadata["anomaly_rate"] = float(
            re.search(r"<anomaly_rate>(.*?)</anomaly_rate>", xml_content).group(1)
        )

        companies = {}
        company_pattern = (
            r'<company name="(.*?)" transactions="(.*?)" total_amount="(.*?)"'
        )
        for match in re.finditer(company_pattern, xml_content):
            companies[match.group(1)] = {
                "transactions": int(match.group(2)),
                "total_amount": float(match.group(3)),
            }

        transactions = []
        transaction_pattern = r'<transaction id="(.*?)" is_anomaly="(.*?)".*?<timestamp>(.*?)</timestamp>.*?<company>(.*?)</company>.*?<amount>(.*?)</amount>'
        matches = list(re.finditer(transaction_pattern, xml_content, re.DOTALL))[:10]

        for match in matches:
            transactions.append(
                {
                    "id": int(match.group(1)),
                    "is_anomaly": match.group(2) == "True",
                    "timestamp": match.group(3),
                    "company": match.group(4),
                    "amount": float(match.group(5)),
                }
            )

        return {
            "metadata": metadata,
            "companies": companies,
            "transactions": transactions,
        }


class AnomalyDetector:

    def __init__(self):
        self.models = {
            "isolation_forest": IsolationForest(contamination=0.05, random_state=42),
            "statistical": None,
        }
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:

        features = ["amount_uah", "hour", "day_of_week", "month", "quarter"]

        df_features = df[features].copy()

        for window in [6, 12, 24]:
            df_features[f"rolling_mean_{window}h"] = (
                df["amount_uah"].rolling(window=window).mean()
            )
            df_features[f"rolling_std_{window}h"] = (
                df["amount_uah"].rolling(window=window).std()
            )

        df_features = df_features.fillna(method="bfill").fillna(method="ffill")

        return df_features.values

    def detect_statistical_anomalies(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> np.ndarray:

        predictions = np.zeros(len(df))

        for hour in range(24):
            hour_data = df[df["hour"] == hour]["amount_uah"]
            if len(hour_data) > 10:
                mean = hour_data.mean()
                std = hour_data.std()
                z_scores = np.abs((hour_data - mean) / std)
                hour_indices = df[df["hour"] == hour].index
                predictions[hour_indices] = (z_scores > threshold).astype(int)

        return predictions

    def fit_and_predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:

        features = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)

        predictions = {}

        self.models["isolation_forest"].fit(features_scaled)
        if_predictions = self.models["isolation_forest"].predict(features_scaled)
        predictions["isolation_forest"] = (if_predictions == -1).astype(int)

        predictions["statistical"] = self.detect_statistical_anomalies(df)

        predictions["ensemble"] = (
            (predictions["isolation_forest"] + predictions["statistical"]) >= 1
        ).astype(int)

        return predictions

    def evaluate_performance(
        self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]
    ) -> Dict:

        results = {}

        for method, y_pred in predictions.items():

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            results[method] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
            }

        return results


class Visualizer:

    @staticmethod
    def plot_time_series_with_anomalies(
        df: pd.DataFrame, predictions: Dict[str, np.ndarray]
    ):

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Оригінальні дані з істинними аномаліями",
                "Isolation Forest",
                "Statistical Method",
                "Ensemble Method",
            ],
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["amount_uah"],
                mode="lines",
                name="Транзакції",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        anomalies = df[df["is_anomaly"] == 1]
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["amount_uah"],
                mode="markers",
                name="Істинні аномалії",
                marker=dict(color="red", size=8),
            ),
            row=1,
            col=1,
        )

        positions = [(1, 2), (2, 1), (2, 2)]
        colors = ["orange", "green", "purple"]

        for i, (method, pred) in enumerate(predictions.items()):
            row, col = positions[i]

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["amount_uah"],
                    mode="lines",
                    name="Транзакції",
                    line=dict(color="blue"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            predicted_anomalies = df[pred == 1]
            fig.add_trace(
                go.Scatter(
                    x=predicted_anomalies["timestamp"],
                    y=predicted_anomalies["amount_uah"],
                    mode="markers",
                    name=f"Передбачені ({method})",
                    marker=dict(color=colors[i], size=8),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            height=800, title_text="Виявлення аномалій у фінансових транзакціях"
        )
        fig.show()

    @staticmethod
    def plot_performance_comparison(results: Dict):

        methods = list(results.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [results[method][metric] for method in methods]
            bars = axes[i].bar(methods, values, alpha=0.7)
            axes[i].set_title(f"{metric.title()}")
            axes[i].set_ylabel("Значення")
            axes[i].set_ylim(0, 1)

            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_data_distribution(df: pd.DataFrame):

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].hist(df["amount_uah"], bins=50, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("Розподіл сум транзакцій")
        axes[0, 0].set_xlabel("Сума (грн)")
        axes[0, 0].set_ylabel("Частота")

        hour_counts = df.groupby("hour")["amount_uah"].mean()
        axes[0, 1].plot(hour_counts.index, hour_counts.values, marker="o")
        axes[0, 1].set_title("Середня сума транзакцій по годинах")
        axes[0, 1].set_xlabel("Година")
        axes[0, 1].set_ylabel("Середня сума (грн)")

        day_counts = df.groupby("day_of_week")["amount_uah"].mean()
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
        axes[0, 2].bar(range(7), day_counts.values, alpha=0.7)
        axes[0, 2].set_title("Середня сума по днях тижня")
        axes[0, 2].set_xticks(range(7))
        axes[0, 2].set_xticklabels(days)
        axes[0, 2].set_ylabel("Середня сума (грн)")

        company_counts = (
            df.groupby("company")["amount_uah"].sum().sort_values(ascending=False)
        )
        axes[1, 0].bar(range(len(company_counts)), company_counts.values, alpha=0.7)
        axes[1, 0].set_title("Загальна сума по компаніях")
        axes[1, 0].set_xticks(range(len(company_counts)))
        axes[1, 0].set_xticklabels(company_counts.index, rotation=45)
        axes[1, 0].set_ylabel("Загальна сума (грн)")

        numeric_cols = ["amount_uah", "hour", "day_of_week", "month", "quarter"]
        corr_matrix = df[numeric_cols].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap="coolwarm", aspect="auto")
        axes[1, 1].set_title("Кореляційна матриця")
        axes[1, 1].set_xticks(range(len(numeric_cols)))
        axes[1, 1].set_yticks(range(len(numeric_cols)))
        axes[1, 1].set_xticklabels(numeric_cols, rotation=45)
        axes[1, 1].set_yticklabels(numeric_cols)
        plt.colorbar(im, ax=axes[1, 1])

        anomaly_by_company = (
            df.groupby("company")["is_anomaly"].mean().sort_values(ascending=False)
        )
        axes[1, 2].bar(
            range(len(anomaly_by_company)),
            anomaly_by_company.values,
            alpha=0.7,
            color="red",
        )
        axes[1, 2].set_title("Частка аномалій по компаніях")
        axes[1, 2].set_xticks(range(len(anomaly_by_company)))
        axes[1, 2].set_xticklabels(anomaly_by_company.index, rotation=45)
        axes[1, 2].set_ylabel("Частка аномалій")

        plt.tight_layout()
        plt.show()


def main():
    print("Виявлення аномалій у фінансових транзакціях енергетичних компаній")
    print("=" * 80)

    print("\n1. ГЕНЕРАЦІЯ ЧАСОВИХ РЯДІВ")
    print("-" * 40)

    generator = EnergyCompanyDataGenerator(seed=42)
    df = generator.generate_time_series(
        start_date="2024-01-01", end_date="2024-03-31", freq="H"
    )

    print(f"Згенеровано {len(df)} записів транзакцій")
    print(f"Період: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"Компанії: {', '.join(df['company'].unique())}")
    print(f"Частка аномалій: {df['is_anomaly'].mean():.3f}")

    print("\n2. ЗБЕРЕЖЕННЯ ДАНИХ У РІЗНИХ ФОРМАТАХ")
    print("-" * 40)

    processor = DataProcessor()

    os.makedirs("data_output", exist_ok=True)

    json_file = "data_output/transactions_nested.json"
    processor.save_to_json(df, json_file, nested=True)
    print(f"Збережено у JSON з вкладеними структурами: {json_file}")

    csv_file = "data_output/transactions.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Збережено у CSV: {csv_file}")

    xml_file = "data_output/transactions.xml"
    processor.save_to_xml(df, xml_file)
    print(f"Збережено у XML: {xml_file}")

    print("\n3. ЗАВАНТАЖЕННЯ ТА ПАРСИНГ СТРУКТУРОВАНИХ ДАНИХ")
    print("-" * 40)

    df_from_json = processor.load_and_parse_json(json_file)
    print(f"Завантажено з JSON: {len(df_from_json)} записів")

    xml_data = processor.parse_xml_with_regex(xml_file)
    print(f"Парсинг XML:")
    print(f"   - Метадані: {xml_data['metadata']['total_records']} записів")
    print(f"   - Компанії: {len(xml_data['companies'])}")
    print(f"   - Частка аномалій: {xml_data['metadata']['anomaly_rate']:.3f}")

    print("\n4. ВИЯВЛЕННЯ АНОМАЛІЙ")
    print("-" * 40)

    detector = AnomalyDetector()
    predictions = detector.fit_and_predict(df)

    print("Результати виявлення аномалій:")
    for method, pred in predictions.items():
        detected_count = np.sum(pred)
        print(f"   - {method}: виявлено {detected_count} аномалій")

    print("\n5. ОЦІНКА ЯКОСТІ ВИЯВЛЕННЯ АНОМАЛІЙ")
    print("-" * 40)

    results = detector.evaluate_performance(df["is_anomaly"].values, predictions)

    print("Метрики якості:")
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"   - Точність (Accuracy): {metrics['accuracy']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - F1-score: {metrics['f1_score']:.3f}")
        print(f"   - True Positives: {metrics['true_positives']}")
        print(f"   - False Positives: {metrics['false_positives']}")
        print(f"   - False Negatives: {metrics['false_negatives']}")

    print("\n6. СТАТИСТИЧНИЙ АНАЛІЗ ДАНИХ")
    print("-" * 40)

    print("Основна статистика:")
    print(f"   - Середня сума транзакції: {df['amount_uah'].mean():.2f} грн")
    print(f"   - Медіана: {df['amount_uah'].median():.2f} грн")
    print(f"   - Стандартне відхилення: {df['amount_uah'].std():.2f} грн")
    print(f"   - Мінімум: {df['amount_uah'].min():.2f} грн")
    print(f"   - Максимум: {df['amount_uah'].max():.2f} грн")

    print("\nСтатистика по компаніях:")
    company_stats = (
        df.groupby("company")
        .agg({"amount_uah": ["count", "mean", "sum"], "is_anomaly": "mean"})
        .round(2)
    )

    for company in df["company"].unique():
        company_data = df[df["company"] == company]
        print(f"   - {company}:")
        print(f"     * Транзакцій: {len(company_data)}")
        print(f"     * Середня сума: {company_data['amount_uah'].mean():.2f} грн")
        print(f"     * Частка аномалій: {company_data['is_anomaly'].mean():.3f}")

    print("\n7. АНАЛІЗ ЧАСОВИХ ПАТЕРНІВ")
    print("-" * 40)

    print("Найактивніші години:")
    hourly_activity = (
        df.groupby("hour")["amount_uah"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )
    for hour in hourly_activity.head(5).index:
        count = hourly_activity.loc[hour, "count"]
        avg_amount = hourly_activity.loc[hour, "mean"]
        print(
            f"   - {hour:02d}:00 - {count} транзакцій, середня сума: {avg_amount:.2f} грн"
        )

    print("\nДні тижня з найбільшою активністю:")
    days = ["Понеділок", "Вівторок", "Середа", "Четвер", "П'ятниця", "Субота", "Неділя"]
    daily_activity = df.groupby("day_of_week")["amount_uah"].agg(["count", "mean"])
    for day_idx in daily_activity.sort_values("count", ascending=False).head(3).index:
        count = daily_activity.loc[day_idx, "count"]
        avg_amount = daily_activity.loc[day_idx, "mean"]
        print(
            f"   - {days[day_idx]} - {count} транзакцій, середня сума: {avg_amount:.2f} грн"
        )

    print("\n8. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
    print("-" * 40)

    visualizer = Visualizer()

    print("Створення графіків розподілу даних...")
    visualizer.plot_data_distribution(df)

    print("Створення графіків порівняння методів...")
    visualizer.plot_performance_comparison(results)

    print("\n9. ЕКСПОРТ РЕЗУЛЬТАТІВ")
    print("-" * 40)

    results_df = df.copy()
    for method, pred in predictions.items():
        results_df[f"predicted_{method}"] = pred

    results_file = "data_output/anomaly_detection_results.csv"
    results_df.to_csv(results_file, index=False, encoding="utf-8")
    print(f"езультати збережено у: {results_file}")

    report = {
        "experiment_info": {
            "dataset_size": len(df),
            "time_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            },
            "companies": df["company"].unique().tolist(),
            "true_anomaly_rate": float(df["is_anomaly"].mean()),
        },
        "performance_metrics": results,
        "data_statistics": {
            "mean_amount": float(df["amount_uah"].mean()),
            "median_amount": float(df["amount_uah"].median()),
            "std_amount": float(df["amount_uah"].std()),
            "min_amount": float(df["amount_uah"].min()),
            "max_amount": float(df["amount_uah"].max()),
        },
        "temporal_patterns": {
            "most_active_hours": hourly_activity.head(5).to_dict(),
            "daily_activity": daily_activity.to_dict(),
        },
    }

    report_file = "data_output/anomaly_detection_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Звіт збережено у: {report_file}")

    print("\n10. ПІДСУМОК ВИКОНАННЯ")
    print("=" * 60)

    best_method = max(results.keys(), key=lambda x: results[x]["f1_score"])
    best_f1 = results[best_method]["f1_score"]

    print(f"\nНайкращий метод: {best_method.upper()}")
    print(f"   F1-score: {best_f1:.3f}")

    print(f"\nСтворено файли:")
    print(f"   - {json_file}")
    print(f"   - {csv_file}")
    print(f"   - {xml_file}")
    print(f"   - {results_file}")
    print(f"   - {report_file}")

    return df, predictions, results


class DataAnalyzer:

    @staticmethod
    def analyze_seasonal_patterns(df: pd.DataFrame):

        monthly_stats = df.groupby("month").agg(
            {"amount_uah": ["mean", "std", "count"], "is_anomaly": "mean"}
        )

        quarterly_stats = df.groupby("quarter").agg(
            {"amount_uah": ["mean", "std", "count"], "is_anomaly": "mean"}
        )

        return {"monthly": monthly_stats, "quarterly": quarterly_stats}

    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 1.5):

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = (series < lower_bound) | (series > upper_bound)

        return outliers

    @staticmethod
    def correlation_analysis(df: pd.DataFrame):

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_columns].corr()

        if "amount_uah" in correlation_matrix.columns:
            amount_correlations = (
                correlation_matrix["amount_uah"].abs().sort_values(ascending=False)
            )
            return {
                "correlation_matrix": correlation_matrix,
                "amount_correlations": amount_correlations,
            }

        return {"correlation_matrix": correlation_matrix}


def demonstrate_advanced_features():
    print("\n" + "=" * 60)
    print("ДОДАТКОВО")
    print("=" * 60)

    generator = EnergyCompanyDataGenerator(seed=123)
    large_df = generator.generate_time_series(
        start_date="2023-01-01", end_date="2024-12-31", freq="H"
    )

    print(f"Згенеровано розширений набір даних: {len(large_df)} записів")

    analyzer = DataAnalyzer()

    seasonal_analysis = analyzer.analyze_seasonal_patterns(large_df)
    print("\nСезонний аналіз:")
    print("Середнє споживання по місяцях:")
    monthly_means = seasonal_analysis["monthly"]["amount_uah"]["mean"]
    for month, mean_amount in monthly_means.items():
        print(f"   Місяць {month}: {mean_amount:.2f} грн")

    correlation_analysis = analyzer.correlation_analysis(large_df)
    print("\nНайсильніші кореляції з сумою транзакцій:")
    if "amount_correlations" in correlation_analysis:
        for variable, correlation in (
            correlation_analysis["amount_correlations"].head(5).items()
        ):
            if variable != "amount_uah":
                print(f"   {variable}: {correlation:.3f}")

    iqr_outliers = analyzer.detect_outliers_iqr(large_df["amount_uah"])
    print(
        f"\nВикиди за методом IQR: {iqr_outliers.sum()} з {len(large_df)} ({iqr_outliers.mean():.3f})"
    )

    return large_df


if __name__ == "__main__":

    df, predictions, results = main()

    large_df = demonstrate_advanced_features()