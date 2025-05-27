import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.interpolate import interp1d, UnivariateSpline, Rbf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from faker import Faker
import time

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
np.random.seed(42)

class DataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = Faker(['uk_UA'])
    
    def generate_time_series_data(self, 
                                start_date: str = "2023-01-01",
                                periods: int = 365,
                                freq: str = "D",
                                missing_rate: float = 0.15) -> pd.DataFrame:
        
        date_index = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        t = np.arange(periods)
        trend = 0.1 * t + 50  
        seasonal = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.sin(2 * np.pi * t / 7)  
        noise = np.random.normal(0, 3, periods)
        temperature = trend + seasonal + noise
        
        growth_rate = np.random.normal(0.001, 0.0002, periods).cumsum()
        revenue = 1000 * np.exp(growth_rate) * (1 + 0.1 * np.random.normal(0, 1, periods))
        
        energy_base = 100 + 50 * np.sin(2 * np.pi * t / 365) * (1 + 0.3 * np.cos(2 * np.pi * t / 7))
        energy_consumption = energy_base * (1 + 0.2 * np.random.normal(0, 1, periods))
        
        production = 200 + np.cumsum(np.random.normal(0, 5, periods))
        
        jump_indices = np.random.choice(periods, size=int(periods * 0.05), replace=False)
        production[jump_indices] += np.random.normal(0, 50, len(jump_indices))
        
        df = pd.DataFrame({
            'date': date_index,
            'temperature': temperature,
            'revenue': revenue,
            'energy_consumption': energy_consumption,
            'production': production
        })
        
        df['season'] = df['date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        df['weekday'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.weekday >= 5
        
        self._introduce_missing_values(df, missing_rate)
        
        return df
    
    def generate_cross_sectional_data(self, n_samples: int = 1000, missing_rate: float = 0.2) -> pd.DataFrame:
        data = []
        for i in range(n_samples):
            
            area = max(30, np.random.gamma(2, 30))  
            rooms = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.25, 0.35, 0.25, 0.05])
            floor = np.random.randint(1, 21)
            age = max(0, np.random.exponential(15))
            
            district = np.random.choice(['Центр', 'Північ', 'Південь', 'Схід', 'Захід'], 
                                      p=[0.15, 0.2, 0.25, 0.2, 0.2])
            building_type = np.random.choice(['Панель', 'Цегла', 'Монолітний'], p=[0.4, 0.35, 0.25])
            
            base_price = 15000  
            
            area_effect = max(area, 50) * 100  
            rooms_effect = rooms * 3000  
            floor_effect = min(floor, 15) * 200  
            age_effect = max(0, 50 - age) * 100  
            
            district_effect = {'Центр': 8000, 'Північ': 2000, 'Південь': -1000, 'Схід': 0, 'Захід': 3000}[district]
            building_effect = {'Панель': -2000, 'Цегла': 1000, 'Монолітний': 4000}[building_type]
            
            price_per_sqm = (base_price + 
                           area_effect/area * 50 + 
                           rooms_effect/area * 20 + 
                           floor_effect + 
                           age_effect + 
                           district_effect + 
                           building_effect)
            
            price_per_sqm *= (1 + np.random.normal(0, 0.05))  
            price_per_sqm = max(8000, price_per_sqm)  
            
            total_price = price_per_sqm * area
            
            metro_distance = max(0.1, np.random.exponential(2))
            park_distance = max(0.1, np.random.exponential(1.5))
            
            
            price_per_sqm -= metro_distance * 300  
            price_per_sqm -= park_distance * 150   
            price_per_sqm = max(8000, price_per_sqm)
            
            data.append({
                'id': i,
                'area': area,
                'rooms': rooms,
                'floor': floor,
                'age': age,
                'district': district,
                'building_type': building_type,
                'price_per_sqm': price_per_sqm,
                'total_price': price_per_sqm * area,
                'metro_distance': metro_distance,
                'park_distance': park_distance,
            })
        
        df = pd.DataFrame(data)
        
        self._introduce_missing_values(df, missing_rate)
        
        return df
    
    def _introduce_missing_values(self, df: pd.DataFrame, missing_rate: float):
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['id']:  
                continue
                
            n_missing = int(len(df) * missing_rate)
            missing_indices = np.random.choice(len(df), size=n_missing, replace=False)
            df.loc[missing_indices, col] = np.nan
    
    def create_missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_patterns = df.copy()
        
        if 'temperature' in df_patterns.columns:
            start_gap = np.random.randint(50, len(df_patterns) - 30)
            df_patterns.loc[start_gap:start_gap+10, 'temperature'] = np.nan
        
        
        if 'revenue' in df_patterns.columns:
            df_patterns.loc[-5:, 'revenue'] = np.nan
        
        
        if 'energy_consumption' in df_patterns.columns:
            df_patterns.loc[:10, 'energy_consumption'] = np.nan
        
        return df_patterns

class InterpolationMethods:
    
    
    def __init__(self):
        self.methods = {}
        self.fitted_models = {}
    
    def linear_interpolation(self, series: pd.Series) -> pd.Series:
        
        return series.interpolate(method='linear')
    
    def polynomial_interpolation(self, series: pd.Series, order: int = 2) -> pd.Series:
        
        return series.interpolate(method='polynomial', order=order)
    
    def spline_interpolation(self, series: pd.Series, order: int = 3) -> pd.Series:
        
        return series.interpolate(method='spline', order=order)
    
    def time_interpolation(self, df: pd.DataFrame, column: str, time_column: str = 'date') -> pd.Series:
        
        try:
            if time_column in df.columns:
                
                temp_df = df.set_index(time_column)[column].copy()
                
                if pd.api.types.is_datetime64_any_dtype(temp_df.index):
                    result = temp_df.interpolate(method='time')
                    
                    return pd.Series(result.values, index=df.index, name=column)
                else:
                    
                    return df[column].interpolate(method='linear')
            else:
                return df[column].interpolate(method='linear')
        except Exception as e:
            print(f"Помилка в часовій інтерполяції: {e}. Використовуємо лінійну.")
            return df[column].interpolate(method='linear')
    
    def forward_fill(self, series: pd.Series) -> pd.Series:
        
        return series.fillna(method='ffill')
    
    def backward_fill(self, series: pd.Series) -> pd.Series:
        
        return series.fillna(method='bfill')
    
    def mean_interpolation(self, series: pd.Series) -> pd.Series:
        
        return series.fillna(series.mean())
    
    def median_interpolation(self, series: pd.Series) -> pd.Series:
        
        return series.fillna(series.median())
    
    def seasonal_interpolation(self, df: pd.DataFrame, column: str, 
                             season_column: str = 'season') -> pd.Series:
        
        series = df[column].copy()
        for season in df[season_column].unique():
            if pd.isna(season):
                continue
            mask = df[season_column] == season
            season_mean = series[mask].mean()
            series.loc[mask] = series.loc[mask].fillna(season_mean)
        
        return series.fillna(series.mean())
    
    def ml_interpolation(self, df: pd.DataFrame, target_column: str, 
                        feature_columns: List[str]) -> pd.Series:

        complete_mask = df[feature_columns + [target_column]].notna().all(axis=1)
        train_data = df[complete_mask]
        
        if len(train_data) < 10:  
            return self.linear_interpolation(df[target_column])
        
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        X_train_processed = pd.get_dummies(X_train)
        model.fit(X_train_processed, y_train)
        
        missing_mask = df[target_column].isna()
        X_missing = df[missing_mask][feature_columns]
        X_missing_processed = pd.get_dummies(X_missing)
        
        missing_cols = set(X_train_processed.columns) - set(X_missing_processed.columns)
        for col in missing_cols:
            X_missing_processed[col] = 0
        X_missing_processed = X_missing_processed[X_train_processed.columns]
        
        predictions = model.predict(X_missing_processed)
        
        result = df[target_column].copy()
        result.loc[missing_mask] = predictions
        
        return result
    
    def cubic_spline_interpolation(self, series: pd.Series) -> pd.Series:
        
        valid_mask = series.notna()
        if valid_mask.sum() < 4:  
            return self.linear_interpolation(series)
        
        x_valid = np.arange(len(series))[valid_mask]
        y_valid = series[valid_mask].values
        
        cs = interp1d(x_valid, y_valid, kind='cubic', fill_value='extrapolate')
        
        x_all = np.arange(len(series))
        y_interpolated = cs(x_all)
        
        result = series.copy()
        result.iloc[:] = y_interpolated
        
        return result
    
    def rbf_interpolation(self, df: pd.DataFrame, target_column: str, 
                         coord_columns: List[str]) -> pd.Series:
        
        try:
            
            complete_mask = df[coord_columns + [target_column]].notna().all(axis=1)
            train_data = df[complete_mask]
            
            if len(train_data) < 5:
                print("RBF: Недостатньо даних для навчання, використовуємо лінійну інтерполяцію")
                return self.linear_interpolation(df[target_column])
            
            coords = train_data[coord_columns].values
            values = train_data[target_column].values
            
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            rbf = Rbf(*coords_scaled.T, values, function='multiquadric', smooth=1.0)
            
            missing_mask = df[target_column].isna()
            if missing_mask.sum() == 0:
                return df[target_column]
                
            missing_coords = df[missing_mask][coord_columns].values
            
            if len(missing_coords) > 0:
                
                missing_coords_scaled = scaler.transform(missing_coords)
                predictions = rbf(*missing_coords_scaled.T)
                
                value_range = values.max() - values.min()
                value_mean = values.mean()
                
                predictions = np.clip(predictions, 
                                    value_mean - 3 * value_range,
                                    value_mean + 3 * value_range)
                
                result = df[target_column].copy()
                result.loc[missing_mask] = predictions
                return result
            
            return df[target_column]
            
        except Exception as e:
            print(f"Помилка в RBF інтерполяції: {e}. Використовуємо лінійну інтерполяцію.")
            return self.linear_interpolation(df[target_column])

class InterpolationEvaluator:
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_method(self, original: pd.Series, interpolated: pd.Series, 
                       missing_mask: pd.Series) -> Dict[str, float]:
        try:
            
            common_index = original.index.intersection(interpolated.index).intersection(missing_mask.index)
            
            orig_aligned = original.reindex(common_index)
            interp_aligned = interpolated.reindex(common_index)
            missing_aligned = missing_mask.reindex(common_index)
            
            orig_missing = orig_aligned[missing_aligned].dropna()
            interp_missing = interp_aligned[missing_aligned].reindex(orig_missing.index).dropna()
            
            common_missing_index = orig_missing.index.intersection(interp_missing.index)
            
            if len(common_missing_index) == 0:
                return {'mse': 1000.0, 'mae': 100.0, 'r2': 0.0, 'mape': 100.0}
            
            orig_values = orig_missing.reindex(common_missing_index)
            interp_values = interp_missing.reindex(common_missing_index)
            
            finite_mask = np.isfinite(orig_values) & np.isfinite(interp_values)
            if finite_mask.sum() == 0:
                return {'mse': 1000.0, 'mae': 100.0, 'r2': 0.0, 'mape': 100.0}
            
            orig_values = orig_values[finite_mask]
            interp_values = interp_values[finite_mask]
            
            mse = mean_squared_error(orig_values, interp_values)
            mae = mean_absolute_error(orig_values, interp_values)
            
            try:
                r2 = r2_score(orig_values, interp_values)
                if np.isnan(r2) or np.isinf(r2):
                    r2 = 0.0
                
                r2 = max(-10.0, min(1.0, r2))
            except:
                r2 = 0.0
            
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    
                    non_zero_mask = np.abs(orig_values) > 1e-8
                    if non_zero_mask.sum() > 0:
                        orig_nz = orig_values[non_zero_mask]
                        interp_nz = interp_values[non_zero_mask]
                        mape_values = np.abs((orig_nz - interp_nz) / orig_nz)
                        mape_values = mape_values[np.isfinite(mape_values)]
                        mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else 100.0
                        
                        mape = min(1000.0, mape)
                    else:
                        mape = 100.0
            except:
                mape = 100.0
            
            mse = min(1000000.0, max(0.0, mse))
            mae = min(10000.0, max(0.0, mae))
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
        except Exception as e:
            print(f"Помилка в оцінці методу: {e}")
            return {'mse': 1000.0, 'mae': 100.0, 'r2': 0.0, 'mape': 100.0}
    
    def compare_methods(self, df: pd.DataFrame, column: str, 
                       methods_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        
        results = []
        original_col = column + '_original'
        
        if original_col not in df.columns:
            print(f"Увага: Колонка {original_col} не знайдена. Використовуємо наявні дані для оцінки.")
            
            original = df[column].interpolate(method='linear')
        else:
            original = df[original_col]
        
        missing_mask = df[column].isna()
        
        for method_name, interpolated in methods_dict.items():
            try:
                metrics = self.evaluate_method(original, interpolated, missing_mask)
                metrics['method'] = method_name
                results.append(metrics)
            except Exception as e:
                print(f"Помилка при оцінці методу {method_name}: {e}")
                
                results.append({
                    'method': method_name,
                    'mse': np.inf,
                    'mae': np.inf,
                    'r2': 0,
                    'mape': np.inf
                })
        
        results_df = pd.DataFrame(results)
        
        results_df = results_df.sort_values(['r2', 'mae'], ascending=[False, True]).reset_index(drop=True)
        
        return results_df

class InterpolationVisualizer:

    def plot_interpolation_comparison(self, df: pd.DataFrame, column: str, 
                                    methods_dict: Dict[str, pd.Series], 
                                    sample_size: int = 100):
        
        if len(df) > sample_size:
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            df_sample = df.iloc[indices].copy()
            methods_sample = {name: series.iloc[indices] for name, series in methods_dict.items()}
        else:
            df_sample = df.copy()
            methods_sample = methods_dict
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        x = range(len(df_sample))
        original = df_sample[column + '_original'] if column + '_original' in df_sample.columns else None
        missing_mask = df_sample[column].isna()
        
        method_names = list(methods_sample.keys())
        
        for i, ax in enumerate(axes):
            
            if original is not None:
                ax.plot(x, original, 'ko-', alpha=0.7, markersize=3, label='Оригінальні дані')
            
            ax.plot(x, df_sample[column], 'ro', alpha=0.7, markersize=4, label='Наявні дані')
            
            start_idx = i * 2
            end_idx = min(start_idx + 2, len(method_names))
            
            colors = ['blue', 'green', 'orange', 'purple']
            
            for j, method_idx in enumerate(range(start_idx, end_idx)):
                if method_idx < len(method_names):
                    method_name = method_names[method_idx]
                    interpolated = methods_sample[method_name]
                    ax.plot(x, interpolated, colors[j], alpha=0.8, linewidth=2, 
                           label=f'{method_name}')
            
            ax.scatter(np.array(x)[missing_mask], 
                      np.full(missing_mask.sum(), ax.get_ylim()[0]), 
                      marker='v', s=20, c='red', alpha=0.7, label='Пропущені значення')
            
            ax.set_title(f'Методи інтерполяції: {column}')
            ax.set_xlabel('Індекс')
            ax.set_ylabel('Значення')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_evaluation_metrics(self, comparison_df: pd.DataFrame):
        
        df_filtered = comparison_df.copy()
        
        for col in ['mse', 'mae', 'mape']:
            infinite_mask = np.isinf(df_filtered[col])
            if infinite_mask.any():
                
                finite_values = df_filtered[col][~infinite_mask]
                if len(finite_values) > 0:
                    max_finite = finite_values.max()
                    df_filtered.loc[infinite_mask, col] = max_finite * 2
                else:
                    df_filtered.loc[infinite_mask, col] = 1000
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['r2', 'mse', 'mae', 'mape']
        titles = ['R² Score (вище = краще)', 'MSE (нижче = краще)', 
                 'MAE (нижче = краще)', 'MAPE, % (нижче = краще)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            
            values = df_filtered[metric]
            
            bars = ax.bar(df_filtered['method'], values, 
                         alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(df_filtered))))
            
            ax.set_title(title)
            ax.set_xlabel('Метод інтерполяції')
            ax.set_ylabel(f'{metric.upper()}')
            ax.tick_params(axis='x', rotation=45)
             
            for bar, value, orig_value in zip(bars, values, comparison_df[metric]):
                height = bar.get_height()
                if np.isfinite(orig_value):
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{orig_value:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           'inf', ha='center', va='bottom', fontsize=9, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_missing_patterns(self, df: pd.DataFrame):

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_matrix = df[numeric_cols].isna()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(missing_matrix.T, cbar=True, yticklabels=True, 
                   cmap='RdYlBu_r', cbar_kws={'label': 'Пропущені значення'})
        plt.title('Паттерни пропущених значень')
        plt.xlabel('Індекс запису')
        plt.ylabel('Змінні')
        plt.tight_layout()
        plt.show()
        
        missing_stats = df[numeric_cols].isna().sum().sort_values(ascending=False)
        missing_pct = (missing_stats / len(df) * 100).round(2)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(missing_stats)), missing_stats.values, alpha=0.7)
        plt.title('Кількість пропущених значень по змінних')
        plt.xlabel('Змінні')
        plt.ylabel('Кількість пропущених значень')
        plt.xticks(range(len(missing_stats)), missing_stats.index, rotation=45)
        
        for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{pct}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    print("Методи інтерполяції для заповнення пропущених значень")
    print("=" * 60)
    
    print("\n1. ГЕНЕРАЦІЯ ТЕСТОВИХ ДАНИХ")
    print("-" * 40)
    
    generator = DataGenerator(seed=42)
    
    ts_data = generator.generate_time_series_data(
        start_date="2023-01-01",
        periods=200,
        freq="D",
        missing_rate=0.2
    )
    
    cs_data = generator.generate_cross_sectional_data(
        n_samples=500,
        missing_rate=0.15
    )
    
    print(f"Створено часові ряди: {ts_data.shape}")
    print(f"Створено кросс-секційні дані: {cs_data.shape}")
    
    print("Генерація оригінальних даних для валідації...")
    original_generator = DataGenerator(seed=123)  
    ts_original = original_generator.generate_time_series_data(
        periods=200, missing_rate=0.0  
    )
    
    for col in ['temperature', 'revenue', 'energy_consumption', 'production']:
        if col in ts_data.columns and col in ts_original.columns:
            ts_data[col + '_original'] = ts_original[col]
    
    print("\n2. АНАЛІЗ ПРОПУЩЕНИХ ЗНАЧЕНЬ")
    print("-" * 40)
    
    visualizer = InterpolationVisualizer()
    
    print("Візуалізація патернів пропущених значень...")
    visualizer.plot_missing_patterns(ts_data)
    
    numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
    missing_info = ts_data[numeric_cols].isna().sum()
    print("\nСтатистика пропущених значень (часові ряди):")
    for col, count in missing_info.items():
        if count > 0:
            pct = count / len(ts_data) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    print("\n3. ЗАСТОСУВАННЯ МЕТОДІВ ІНТЕРПОЛЯЦІЇ")
    print("-" * 40)
    
    interpolator = InterpolationMethods()
    
    test_column = 'temperature'
    print(f"\nТестування методів на колонці '{test_column}':")
    
    start_time = time.time()
    
    methods_results = {}
    
    print("  - Лінійна інтерполяція...")
    methods_results['Лінійна'] = interpolator.linear_interpolation(ts_data[test_column])
    
    print("  - Поліноміальна інтерполяція...")
    methods_results['Поліноміальна'] = interpolator.polynomial_interpolation(ts_data[test_column], order=2)
    
    print("  - Сплайн інтерполяція...")
    methods_results['Сплайн'] = interpolator.spline_interpolation(ts_data[test_column])
    
    print("  - Часова інтерполяція...")
    methods_results['Часова'] = interpolator.time_interpolation(ts_data, test_column)
    
    print("  - Forward Fill...")
    methods_results['Forward Fill'] = interpolator.forward_fill(ts_data[test_column])
    
    print("  - Backward Fill...")
    methods_results['Backward Fill'] = interpolator.backward_fill(ts_data[test_column])
    
    print("  - Середнє значення...")
    methods_results['Середнє'] = interpolator.mean_interpolation(ts_data[test_column])
    
    print("  - Медіана...")
    methods_results['Медіана'] = interpolator.median_interpolation(ts_data[test_column])
    
    print("  - Сезонна інтерполяція...")
    methods_results['Сезонна'] = interpolator.seasonal_interpolation(ts_data, test_column)
    
    print("  - Кубічний сплайн...")
    methods_results['Кубічний сплайн'] = interpolator.cubic_spline_interpolation(ts_data[test_column])
    
    print("  - Машинне навчання...")
    if 'energy_consumption' in ts_data.columns and 'production' in ts_data.columns:
        feature_cols = ['energy_consumption', 'production']
        
        for col in feature_cols:
            ts_data[col] = interpolator.linear_interpolation(ts_data[col])
        
        methods_results['ML (Random Forest)'] = interpolator.ml_interpolation(
            ts_data, test_column, feature_cols
        )
    
    processing_time = time.time() - start_time
    print(f"\nЧас обробки: {processing_time:.2f} секунд")
    
    print("\n4. ОЦІНКА ЯКОСТІ МЕТОДІВ")
    print("-" * 40)
    
    evaluator = InterpolationEvaluator()
    comparison_results = evaluator.compare_methods(ts_data, test_column, methods_results)
    
    print("Результати порівняння методів:")
    print(comparison_results.round(4))
    
    best_method = comparison_results.iloc[0]['method']
    best_r2 = comparison_results.iloc[0]['r2']
    print(f"\nНайкращий метод: {best_method} (R² = {best_r2:.4f})")
    
    print("\n5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
    print("-" * 40)
    
    print("Створення графіків порівняння методів...")
    visualizer.plot_interpolation_comparison(ts_data, test_column, methods_results)
    
    print("Створення графіків метрик...")
    visualizer.plot_evaluation_metrics(comparison_results)
    
    print("\n6. ТЕСТУВАННЯ НА КРОСС-СЕКЦІЙНИХ ДАНИХ")
    print("-" * 40)
    
    cs_test_column = 'price_per_sqm'
    print(f"Тестування методів на колонці '{cs_test_column}':")
    
    cs_data[cs_test_column + '_original'] = generator.generate_cross_sectional_data(
        n_samples=500, missing_rate=0.0
    )[cs_test_column]
    
    cs_methods_results = {}
    
    cs_methods_results['Лінійна'] = interpolator.linear_interpolation(cs_data[cs_test_column])
    cs_methods_results['Середнє'] = interpolator.mean_interpolation(cs_data[cs_test_column])
    cs_methods_results['Медіана'] = interpolator.median_interpolation(cs_data[cs_test_column])
    
    feature_cols_cs = ['area', 'rooms', 'floor', 'age', 'metro_distance']
    
    for col in feature_cols_cs:
        if col in cs_data.columns:
            cs_data[col] = interpolator.linear_interpolation(cs_data[col])
    
    cs_methods_results['ML (Random Forest)'] = interpolator.ml_interpolation(
        cs_data, cs_test_column, feature_cols_cs
    )
    
    if 'metro_distance' in cs_data.columns and 'park_distance' in cs_data.columns:
        cs_data['park_distance'] = interpolator.linear_interpolation(cs_data['park_distance'])
        cs_methods_results['RBF'] = interpolator.rbf_interpolation(
            cs_data, cs_test_column, ['metro_distance', 'park_distance']
        )
    
    cs_comparison = evaluator.compare_methods(cs_data, cs_test_column, cs_methods_results)
    print("\nРезультати для кросс-секційних даних:")
    print(cs_comparison.round(4))
    
    print("\n7.ВИСНОВКИ")
    print("-" * 40)
    
    print("РЕЗУЛЬТАТИ АНАЛІЗУ:")
    print(f"   - Найкращий метод для часових рядів: {best_method}")
    print(f"   - R² для часових рядів: {best_r2:.4f}")
    
    if not cs_comparison.empty:
        cs_best_method = cs_comparison.iloc[0]['method']
        cs_best_r2 = cs_comparison.iloc[0]['r2']
        print(f"   - Найкращий метод для кросс-секційних даних: {cs_best_method}")
        print(f"   - R² для кросс-секційних даних: {cs_best_r2:.4f}")
    
    print("\n8. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print("-" * 40)
    
    comparison_results.to_csv('interpolation_comparison_timeseries.csv', index=False)
    cs_comparison.to_csv('interpolation_comparison_crosssectional.csv', index=False)
    
    ts_results_df = ts_data[[test_column, test_column + '_original']].copy()
    for method_name, result in methods_results.items():
        ts_results_df[f'{method_name}_interpolated'] = result
    
    ts_results_df.to_csv('interpolated_timeseries_data.csv', index=False)
    
    print("Результати збережено:")
    print("   - interpolation_comparison_timeseries.csv")
    print("   - interpolation_comparison_crosssectional.csv") 
    print("   - interpolated_timeseries_data.csv")
    
    print("\n9. ПІДСУМОК")
    print("=" * 50)
    
    print("Успішно виконано:")
    
    print(f"\nНайефективніші методи:")
    print(f"   - Часові ряди: {best_method}")
    if not cs_comparison.empty:
        print(f"   - Кросс-секційні дані: {cs_best_method}")
      
    return ts_data, cs_data, comparison_results, cs_comparison

if __name__ == "__main__":
    ts_data, cs_data, ts_results, cs_results = main()