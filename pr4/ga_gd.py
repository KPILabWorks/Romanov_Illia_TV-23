import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import time
import warnings
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
np.random.seed(42)
random.seed(42)

@dataclass
class TariffStructure:
    
    base_rate: float  
    peak_multiplier: float  
    off_peak_multiplier: float  
    seasonal_summer_multiplier: float  
    seasonal_winter_multiplier: float  
    volume_discount_threshold: float  
    volume_discount_rate: float  
    
    def __post_init__(self):
        
        self.base_rate = max(0.1, min(10.0, self.base_rate))
        self.peak_multiplier = max(1.0, min(3.0, self.peak_multiplier))
        self.off_peak_multiplier = max(0.5, min(1.0, self.off_peak_multiplier))
        self.seasonal_summer_multiplier = max(0.8, min(1.2, self.seasonal_summer_multiplier))
        self.seasonal_winter_multiplier = max(0.8, min(1.5, self.seasonal_winter_multiplier))
        self.volume_discount_threshold = max(100, min(10000, self.volume_discount_threshold))
        self.volume_discount_rate = max(0.0, min(0.3, self.volume_discount_rate))
    
    def to_array(self) -> np.ndarray:
        
        return np.array([
            self.base_rate,
            self.peak_multiplier,
            self.off_peak_multiplier,
            self.seasonal_summer_multiplier,
            self.seasonal_winter_multiplier,
            self.volume_discount_threshold,
            self.volume_discount_rate
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TariffStructure':
        
        return cls(*arr)

class EnergyConsumerSimulator:
    
    def __init__(self, n_consumers: int = 200, n_days: int = 90):  
        self.n_consumers = n_consumers
        self.n_days = n_days
        self.consumers_data = self._generate_consumers()
        self.consumption_data = self._generate_consumption_patterns()
    
    def _generate_consumers(self) -> pd.DataFrame:
        
        consumer_types = ['residential', 'commercial', 'industrial']
        type_weights = [0.7, 0.2, 0.1]
        
        consumers = []
        for i in range(self.n_consumers):
            consumer_type = np.random.choice(consumer_types, p=type_weights)
            
            if consumer_type == 'residential':
                base_consumption = np.random.normal(300, 100)  
                price_sensitivity = np.random.normal(0.8, 0.2)
            elif consumer_type == 'commercial':
                base_consumption = np.random.normal(2000, 500)
                price_sensitivity = np.random.normal(0.6, 0.15)
            else:  
                base_consumption = np.random.normal(10000, 3000)
                price_sensitivity = np.random.normal(0.4, 0.1)
            
            consumers.append({
                'consumer_id': i,
                'type': consumer_type,
                'base_consumption': max(50, base_consumption),
                'price_sensitivity': max(0.1, min(1.0, price_sensitivity)),
                'peak_usage_tendency': np.random.beta(2, 5),  
                'seasonal_variation': np.random.normal(1.0, 0.1)
            })
        
        return pd.DataFrame(consumers)
    
    def _generate_consumption_patterns(self) -> pd.DataFrame:
        
        dates = pd.date_range('2024-01-01', periods=self.n_days, freq='D')
        
        consumption_data = []
        for _, consumer in self.consumers_data.iterrows():
            for date in dates:
                
                month = date.month
                if month in [6, 7, 8]:  
                    seasonal_factor = 1.2  
                elif month in [12, 1, 2]:  
                    seasonal_factor = 1.4  
                else:
                    seasonal_factor = 1.0
                
                seasonal_factor *= consumer['seasonal_variation']
                
                daily_base = consumer['base_consumption'] / 30  
                
                peak_hours_consumption = daily_base * 0.4 * (1 + consumer['peak_usage_tendency'])
                
                off_peak_consumption = daily_base * 0.6 * (1 - consumer['peak_usage_tendency'] * 0.3)
                
                total_daily = (peak_hours_consumption + off_peak_consumption) * seasonal_factor
                
                total_daily *= np.random.normal(1.0, 0.1)
                
                consumption_data.append({
                    'consumer_id': consumer['consumer_id'],
                    'date': date,
                    'month': month,
                    'is_summer': month in [6, 7, 8],
                    'is_winter': month in [12, 1, 2],
                    'peak_consumption': peak_hours_consumption * seasonal_factor,
                    'off_peak_consumption': off_peak_consumption * seasonal_factor,
                    'total_consumption': total_daily,
                    'consumer_type': consumer['type'],
                    'price_sensitivity': consumer['price_sensitivity']
                })
        
        return pd.DataFrame(consumption_data)

class TariffOptimizer:
    
    def __init__(self, simulator: EnergyConsumerSimulator):
        self.simulator = simulator
        self.consumption_data = simulator.consumption_data
        self.consumers_data = simulator.consumers_data
        
        self.target_revenue = self._calculate_target_revenue()
        self.social_welfare_weight = 0.4  
        self.revenue_weight = 0.6  
        
        self._sample_data = self._prepare_sample_data()
    
    def _prepare_sample_data(self):
        
        sample_size = min(30, self.simulator.n_consumers)
        sample_consumers = np.random.choice(
            self.consumption_data['consumer_id'].unique(), 
            sample_size, 
            replace=False
        )
        
        sample_data = []
        for consumer_id in sample_consumers:
            consumer_data = self.consumption_data[
                self.consumption_data['consumer_id'] == consumer_id
            ].sample(n=min(20, len(self.consumption_data[self.consumption_data['consumer_id'] == consumer_id])))
            sample_data.append(consumer_data)
        
        return sample_data
    
    def _calculate_target_revenue(self) -> float:
        
        total_consumption = self.consumption_data['total_consumption'].sum()
        
        return total_consumption * 2.5  
    
    def calculate_bill(self, tariff: TariffStructure, consumption_row: pd.Series) -> float:
        
        peak_cost = consumption_row['peak_consumption'] * tariff.base_rate * tariff.peak_multiplier
        off_peak_cost = consumption_row['off_peak_consumption'] * tariff.base_rate * tariff.off_peak_multiplier
        
        
        if consumption_row['is_summer']:
            seasonal_multiplier = tariff.seasonal_summer_multiplier
        elif consumption_row['is_winter']:
            seasonal_multiplier = tariff.seasonal_winter_multiplier
        else:
            seasonal_multiplier = 1.0
        
        base_bill = (peak_cost + off_peak_cost) * seasonal_multiplier
        
        total_consumption = consumption_row['total_consumption']
        if total_consumption > tariff.volume_discount_threshold / 365:  
            discount = base_bill * tariff.volume_discount_rate
            base_bill -= discount
        
        return max(0.01, base_bill)  
    
    def calculate_demand_response(self, tariff: TariffStructure, consumption_row: pd.Series) -> float:
        
        effective_rate = tariff.base_rate * tariff.peak_multiplier
        price_change_factor = 1 - consumption_row['price_sensitivity'] * 0.1 * (effective_rate - 3.0)
        return max(0.3, min(1.5, price_change_factor))  
    
    def objective_function(self, tariff: TariffStructure) -> float:
        
        try:
            total_revenue = 0
            total_social_welfare = 0
            
            for consumer_data in self._sample_data:
                annual_bill = 0
                annual_consumption = 0
                
                for _, row in consumer_data.iterrows():
                    
                    bill = self.calculate_bill(tariff, row)
                    
                    demand_factor = self.calculate_demand_response(tariff, row)
                    adjusted_bill = bill * demand_factor
                    adjusted_consumption = row['total_consumption'] * demand_factor
                    
                    annual_bill += adjusted_bill
                    annual_consumption += adjusted_consumption
                
                annual_bill *= (365 / len(consumer_data))
                total_revenue += annual_bill
                
                consumer_type = consumer_data.iloc[0]['consumer_type']
                if consumer_type == 'residential':
                    welfare_weight = 1.0
                elif consumer_type == 'commercial':
                    welfare_weight = 0.7
                else:
                    welfare_weight = 0.5
                
                
                normalized_bill = annual_bill / 10000  
                social_welfare = welfare_weight * max(0, 1 - normalized_bill)
                total_social_welfare += social_welfare
            
            scale_factor = (self.simulator.n_consumers / len(self._sample_data))
            total_revenue *= scale_factor
            total_social_welfare *= scale_factor
            
            revenue_penalty = abs(total_revenue - self.target_revenue) / self.target_revenue
            
            if total_revenue < self.target_revenue * 0.8:
                revenue_penalty += 0.5
            
            normalized_social_welfare = total_social_welfare / self.simulator.n_consumers
            
            objective = (self.revenue_weight * revenue_penalty - 
                        self.social_welfare_weight * normalized_social_welfare)
            
            if np.isnan(objective) or np.isinf(objective):
                return 1000.0  
                
            return float(objective)
            
        except Exception as e:
            print(f"Помилка в objective_function: {e}")
            return 1000.0  

class GeneticAlgorithmOptimizer(TariffOptimizer):
    
    def __init__(self, simulator: EnergyConsumerSimulator, 
                 population_size: int = 20, generations: int = 30,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        super().__init__(simulator)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        
        self.bounds = [
            (0.5, 8.0),    
            (1.0, 3.0),    
            (0.5, 1.0),    
            (0.8, 1.2),    
            (0.8, 1.5),    
            (100, 5000),   
            (0.0, 0.3)     
        ]
        
        self.history = []
    
    def create_individual(self) -> np.ndarray:
        
        individual = []
        for (low, high) in self.bounds:
            individual.append(np.random.uniform(low, high))
        return np.array(individual)
    
    def create_population(self) -> List[np.ndarray]:
        
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual: np.ndarray) -> float:
        
        tariff = TariffStructure.from_array(individual)
        return -self.objective_function(tariff)  
    
    def selection(self, population: List[np.ndarray], fitnesses: List[float]) -> List[np.ndarray]:
        
        selected = []
        for _ in range(self.population_size):
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
        return selected
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        
        for i, (low, high) in enumerate(self.bounds):
            child1[i] = np.clip(child1[i], low, high)
            child2[i] = np.clip(child2[i], low, high)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                
                mutation = np.random.normal(0, (high - low) * 0.1)
                mutated[i] = np.clip(mutated[i] + mutation, low, high)
        return mutated
    
    def optimize(self) -> Tuple[TariffStructure, List[Dict]]:
        
        print("Запуск генетичного алгоритму...")
        start_time = time.time()
        
        population = self.create_population()
        
        for generation in range(self.generations):
            
            fitnesses = [self.fitness(individual) for individual in population]
            
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            self.history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_individual': population[np.argmax(fitnesses)].copy()
            })
            
            if generation % 10 == 0 or generation < 5:
                print(f"Покоління {generation}: найкраща пристосованість = {best_fitness:.4f}")
            
            selected = self.selection(population, fitnesses)
            
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        final_fitnesses = [self.fitness(individual) for individual in population]
        best_individual = population[np.argmax(final_fitnesses)]
        best_tariff = TariffStructure.from_array(best_individual)
        
        optimization_time = time.time() - start_time
        print(f"Генетичний алгоритм завершено за {optimization_time:.2f} секунд")
        
        return best_tariff, self.history

class GradientDescentOptimizer(TariffOptimizer):
    
    def __init__(self, simulator: EnergyConsumerSimulator):
        super().__init__(simulator)
        self.history = []
        self.iteration_count = 0
    
    def objective_wrapper(self, x: np.ndarray) -> float:
        
        try:
            
            bounds = [
                (0.5, 8.0),    # base_rate
                (1.0, 3.0),    # peak_multiplier
                (0.5, 1.0),    # off_peak_multiplier
                (0.8, 1.2),    # seasonal_summer_multiplier
                (0.8, 1.5),    # seasonal_winter_multiplier
                (100, 5000),   # volume_discount_threshold
                (0.0, 0.3)     # volume_discount_rate
            ]
            
            x_clipped = x.copy()
            for i, (low, high) in enumerate(bounds):
                x_clipped[i] = np.clip(x_clipped[i], low, high)
            
            tariff = TariffStructure.from_array(x_clipped)
            obj_value = self.objective_function(tariff)
            
            if np.isnan(obj_value) or np.isinf(obj_value):
                obj_value = 1000.0
            
            self.history.append({
                'iteration': self.iteration_count,
                'objective': obj_value,
                'parameters': x_clipped.copy()
            })
            
            self.iteration_count += 1
            
            if self.iteration_count % 20 == 0:
                print(f"Ітерація {self.iteration_count}: цільова функція = {obj_value:.6f}")
            
            return float(obj_value)
            
        except Exception as e:
            print(f"Помилка в objective_wrapper: {e}")
            return 1000.0
    
    def optimize(self) -> Tuple[TariffStructure, List[Dict]]:
        
        print("Запуск градієнтного спуску...")
        start_time = time.time()
        
        x0 = np.array([2.5, 1.5, 0.8, 1.0, 1.1, 1500, 0.15])
        
        bounds = [
            (0.5, 8.0),    # base_rate
            (1.0, 3.0),    # peak_multiplier
            (0.5, 1.0),    # off_peak_multiplier
            (0.8, 1.2),    # seasonal_summer_multiplier
            (0.8, 1.5),    # seasonal_winter_multiplier
            (100, 5000),   # volume_discount_threshold
            (0.0, 0.3)     # volume_discount_rate
        ]
        
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_result = None
        best_objective = float('inf')
        
        for method in methods:
            try:
                print(f"Спробуємо метод: {method}")
                self.iteration_count = 0
                self.history = []
                
                if method == 'L-BFGS-B':
                    options = {'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
                elif method == 'TNC':
                    options = {'maxiter': 200, 'ftol': 1e-6, 'xtol': 1e-6}
                else:  # SLSQP
                    options = {'maxiter': 200, 'ftol': 1e-6}
                
                result = minimize(
                    self.objective_wrapper,
                    x0,
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    print(f"Метод {method} успішний! Цільова функція: {result.fun:.6f}")
                    break
                else:
                    print(f"Метод {method} не вдався: {result.message}")
                    
            except Exception as e:
                print(f"Помилка з методом {method}: {e}")
                continue
        
        if best_result is None or not best_result.success:
            print("Стандартні методи не вдалися. Використовуємо власний градієнтний спуск...")
            best_result = self._custom_gradient_descent(x0, bounds)
        
        optimization_time = time.time() - start_time
        print(f"Градієнтний спуск завершено за {optimization_time:.2f} секунд")
        
        if best_result is not None:
            print(f"Успішність оптимізації: {getattr(best_result, 'success', True)}")
            print(f"Кількість ітерацій: {len(self.history)}")
            best_tariff = TariffStructure.from_array(best_result.x)
        else:
            print("Оптимізація не вдалася, використовуємо початкові значення")
            best_tariff = TariffStructure.from_array(x0)
        
        return best_tariff, self.history
    
    def _custom_gradient_descent(self, x0, bounds, learning_rate=0.01, max_iterations=100):
        
        x = x0.copy()
        
        class SimpleResult:
            def __init__(self, x, fun, success, nit):
                self.x = x
                self.fun = fun
                self.success = success
                self.nit = nit
        
        for i in range(max_iterations):
            
            current_obj = self.objective_wrapper(x)
            gradient = np.zeros_like(x)
            
            eps = 1e-6
            for j in range(len(x)):
                x_plus = x.copy()
                x_plus[j] += eps
                x_minus = x.copy()
                x_minus[j] -= eps
                
                obj_plus = self.objective_wrapper(x_plus)
                obj_minus = self.objective_wrapper(x_minus)
                
                gradient[j] = (obj_plus - obj_minus) / (2 * eps)
            
            x_new = x - learning_rate * gradient
            
            for j, (low, high) in enumerate(bounds):
                x_new[j] = np.clip(x_new[j], low, high)
            
            
            if np.linalg.norm(x_new - x) < 1e-6:
                print(f"Збіжність досягнута на ітерації {i}")
                break
            
            x = x_new
            
            if i % 10 == 0:
                print(f"Власний ГС ітерація {i}: цільова функція = {current_obj:.6f}")
        final_obj = self.objective_wrapper(x)
        return SimpleResult(x, final_obj, True, i + 1)

class ResultsAnalyzer:
    
    
    def __init__(self, ga_tariff: TariffStructure, ga_history: List[Dict],
                 gd_tariff: TariffStructure, gd_history: List[Dict],
                 simulator: EnergyConsumerSimulator):
        self.ga_tariff = ga_tariff
        self.ga_history = ga_history
        self.gd_tariff = gd_tariff
        self.gd_history = gd_history
        self.simulator = simulator
        self.optimizer = TariffOptimizer(simulator)
    
    def compare_tariffs(self) -> Dict:
        
        ga_objective = self.optimizer.objective_function(self.ga_tariff)
        gd_objective = self.optimizer.objective_function(self.gd_tariff)
        
        comparison = {
            'genetic_algorithm': {
                'tariff': self.ga_tariff,
                'objective_value': ga_objective,
                'convergence_time': len(self.ga_history),
                'final_fitness': self.ga_history[-1]['best_fitness'] if self.ga_history else 0
            },
            'gradient_descent': {
                'tariff': self.gd_tariff,
                'objective_value': gd_objective,
                'convergence_time': len(self.gd_history),
                'iterations': len(self.gd_history)
            }
        }
        return comparison
    
    def calculate_financial_metrics(self, tariff: TariffStructure) -> Dict:
        
        total_revenue = 0
        bills_by_type = {'residential': [], 'commercial': [], 'industrial': []}
        
        for consumer_id in self.simulator.consumption_data['consumer_id'].unique():
            consumer_data = self.simulator.consumption_data[
                self.simulator.consumption_data['consumer_id'] == consumer_id
            ]
            
            annual_bill = 0
            for _, row in consumer_data.iterrows():
                bill = self.optimizer.calculate_bill(tariff, row)
                demand_factor = self.optimizer.calculate_demand_response(tariff, row)
                annual_bill += bill * demand_factor
            
            total_revenue += annual_bill
            consumer_type = consumer_data.iloc[0]['consumer_type']
            bills_by_type[consumer_type].append(annual_bill)
        
        return {
            'total_revenue': total_revenue,
            'avg_residential_bill': np.mean(bills_by_type['residential']) if bills_by_type['residential'] else 0,
            'avg_commercial_bill': np.mean(bills_by_type['commercial']) if bills_by_type['commercial'] else 0,
            'avg_industrial_bill': np.mean(bills_by_type['industrial']) if bills_by_type['industrial'] else 0,
            'revenue_per_kwh': total_revenue / max(1, self.simulator.consumption_data['total_consumption'].sum())
        }
    
    def plot_convergence_comparison(self):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if self.ga_history:
            generations = [h['generation'] for h in self.ga_history]
            best_fitnesses = [h['best_fitness'] for h in self.ga_history]
            avg_fitnesses = [h['avg_fitness'] for h in self.ga_history]
            
            ax1.plot(generations, best_fitnesses, 'b-', label='Найкраща пристосованість', linewidth=2)
            ax1.plot(generations, avg_fitnesses, 'b--', label='Середня пристосованість', alpha=0.7)
            ax1.set_title('Генетичний алгоритм')
            ax1.set_xlabel('Покоління')
            ax1.set_ylabel('Пристосованість')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if self.gd_history:
            iterations = [h['iteration'] for h in self.gd_history]
            objectives = [-h['objective'] for h in self.gd_history]  
            
            ax2.plot(iterations, objectives, 'r-', label='Цільова функція', linewidth=2)
            ax2.set_title('Градієнтний спуск')
            ax2.set_xlabel('Ітерація')
            ax2.set_ylabel('Пристосованість (інвертована)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_tariff_comparison(self):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        params = ['Базова ставка', 'Піковий множник', 'Непіковий множник', 
                 'Літній множник', 'Зимовий множник', 'Поріг знижки', 'Розмір знижки']
        
        ga_values = self.ga_tariff.to_array()
        gd_values = self.gd_tariff.to_array()
        
        ga_normalized = []
        gd_normalized = []
        bounds = [(0.5, 8.0), (1.0, 3.0), (0.5, 1.0), (0.8, 1.2), 
                 (0.8, 1.5), (100, 5000), (0.0, 0.3)]
        
        for i, (low, high) in enumerate(bounds):
            ga_norm = (ga_values[i] - low) / (high - low)
            gd_norm = (gd_values[i] - low) / (high - low)
            ga_normalized.append(ga_norm)
            gd_normalized.append(gd_norm)
        
        x = np.arange(len(params))
        width = 0.35
        
        ax1.bar(x - width/2, ga_normalized, width, label='Генетичний алгоритм', alpha=0.7)
        ax1.bar(x + width/2, gd_normalized, width, label='Градієнтний спуск', alpha=0.7)
        ax1.set_xlabel('Параметри тарифу')
        ax1.set_ylabel('Нормалізовані значення')
        ax1.set_title('Порівняння структури тарифів')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        ga_metrics = self.calculate_financial_metrics(self.ga_tariff)
        gd_metrics = self.calculate_financial_metrics(self.gd_tariff)
        
        metrics_names = ['Загальний дохід (млн грн)', 'Середній рахунок\n(побутові)', 
                        'Середній рахунок\n(комерційні)', 'Середній рахунок\n(промислові)']
        
        ga_metrics_values = [
            ga_metrics['total_revenue'] / 1e6,
            ga_metrics['avg_residential_bill'],
            ga_metrics['avg_commercial_bill'],
            ga_metrics['avg_industrial_bill']
        ]
        
        gd_metrics_values = [
            gd_metrics['total_revenue'] / 1e6,
            gd_metrics['avg_residential_bill'],
            gd_metrics['avg_commercial_bill'],
            gd_metrics['avg_industrial_bill']
        ]
        
        x2 = np.arange(len(metrics_names))
        ax2.bar(x2 - width/2, ga_metrics_values, width, label='Генетичний алгоритм', alpha=0.7)
        ax2.bar(x2 + width/2, gd_metrics_values, width, label='Градієнтний спуск', alpha=0.7)
        ax2.set_xlabel('Фінансові метрики')
        ax2.set_ylabel('Значення')
        ax2.set_title('Порівняння фінансових результатів')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metrics_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_consumer_impact_analysis(self):
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        consumer_types = ['residential', 'commercial', 'industrial']
        ga_bills = {ctype: [] for ctype in consumer_types}
        gd_bills = {ctype: [] for ctype in consumer_types}
        
        for consumer_id in self.simulator.consumption_data['consumer_id'].unique():
            consumer_data = self.simulator.consumption_data[
                self.simulator.consumption_data['consumer_id'] == consumer_id
            ]
            
            consumer_type = consumer_data.iloc[0]['consumer_type']
            
            ga_annual_bill = 0
            gd_annual_bill = 0
            
            for _, row in consumer_data.iterrows():
                ga_bill = self.optimizer.calculate_bill(self.ga_tariff, row)
                gd_bill = self.optimizer.calculate_bill(self.gd_tariff, row)
                
                ga_demand_factor = self.optimizer.calculate_demand_response(self.ga_tariff, row)
                gd_demand_factor = self.optimizer.calculate_demand_response(self.gd_tariff, row)
                
                ga_annual_bill += ga_bill * ga_demand_factor
                gd_annual_bill += gd_bill * gd_demand_factor
            
            ga_bills[consumer_type].append(ga_annual_bill)
            gd_bills[consumer_type].append(gd_annual_bill)
        
        for i, ctype in enumerate(consumer_types):
            row = i // 2
            col = i % 2
            
            if ga_bills[ctype] and gd_bills[ctype]:  
                axes[row, col].hist(ga_bills[ctype], bins=20, alpha=0.7, label='Генетичний алгоритм', density=True)
                axes[row, col].hist(gd_bills[ctype], bins=20, alpha=0.7, label='Градієнтний спуск', density=True)
            axes[row, col].set_title(f'Розподіл рахунків: {ctype}')
            axes[row, col].set_xlabel('Річний рахунок (грн)')
            axes[row, col].set_ylabel('Щільність')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        avg_ga = [np.mean(ga_bills[ctype]) if ga_bills[ctype] else 0 for ctype in consumer_types]
        avg_gd = [np.mean(gd_bills[ctype]) if gd_bills[ctype] else 0 for ctype in consumer_types]
        
        x = np.arange(len(consumer_types))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, avg_ga, width, label='Генетичний алгоритм', alpha=0.7)
        axes[1, 1].bar(x + width/2, avg_gd, width, label='Градієнтний спуск', alpha=0.7)
        axes[1, 1].set_xlabel('Тип споживача')
        axes[1, 1].set_ylabel('Середній річний рахунок (грн)')
        axes[1, 1].set_title('Порівняння середніх рахунків')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([t.title() for t in consumer_types])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        
        for i, (ga_val, gd_val) in enumerate(zip(avg_ga, avg_gd)):
            if ga_val > 0:
                axes[1, 1].text(i - width/2, ga_val + max(max(avg_ga), max(avg_gd)) * 0.01, f'{ga_val:.0f}', 
                               ha='center', va='bottom', fontsize=9)
            if gd_val > 0:
                axes[1, 1].text(i + width/2, gd_val + max(max(avg_ga), max(avg_gd)) * 0.01, f'{gd_val:.0f}', 
                               ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_analysis(self):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        monthly_data = self.simulator.consumption_data.groupby('month').agg({
            'total_consumption': 'mean',
            'peak_consumption': 'mean',
            'off_peak_consumption': 'mean'
        }).reset_index()
        
        
        all_months = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер', 
                     'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
        
        available_months = [all_months[i-1] for i in monthly_data['month'].values]
        
        ga_monthly_costs = []
        gd_monthly_costs = []
        
        for _, month_row in monthly_data.iterrows():
            month = month_row['month']
            month_data = self.simulator.consumption_data[
                self.simulator.consumption_data['month'] == month
            ]
            
            ga_cost = 0
            gd_cost = 0
            count = 0
            
            for _, row in month_data.iterrows():
                ga_bill = self.optimizer.calculate_bill(self.ga_tariff, row)
                gd_bill = self.optimizer.calculate_bill(self.gd_tariff, row)
                ga_cost += ga_bill
                gd_cost += gd_bill
                count += 1
            
            ga_monthly_costs.append(ga_cost / max(1, count))
            gd_monthly_costs.append(gd_cost / max(1, count))
        
        ax1.plot(available_months, ga_monthly_costs, 'o-', label='Генетичний алгоритм', linewidth=2, markersize=8)
        ax1.plot(available_months, gd_monthly_costs, 's-', label='Градієнтний спуск', linewidth=2, markersize=8)
        ax1.set_title('Сезонні зміни середньої вартості')
        ax1.set_xlabel('Місяць')
        ax1.set_ylabel('Середня денна вартість (грн)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(available_months, monthly_data['total_consumption'], alpha=0.7, color='skyblue')
        ax2.set_title('Середнє споживання по місяцях')
        ax2.set_xlabel('Місяць')
        ax2.set_ylabel('Споживання (кВт*год)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self) -> str:
        
        ga_metrics = self.calculate_financial_metrics(self.ga_tariff)
        gd_metrics = self.calculate_financial_metrics(self.gd_tariff)
        comparison = self.compare_tariffs()
        
        report = f"""
ДЕТАЛЬНИЙ ЗВІТ ПОРІВНЯННЯ МЕТОДІВ ОПТИМІЗАЦІЇ ТАРИФІВ
=====================================================

1. ПАРАМЕТРИ ОПТИМІЗОВАНИХ ТАРИФІВ
---------------------------------

Генетичний алгоритм:
  - Базова ставка: {self.ga_tariff.base_rate:.3f} грн/кВт*год
  - Піковий множник: {self.ga_tariff.peak_multiplier:.3f}
  - Непіковий множник: {self.ga_tariff.off_peak_multiplier:.3f}
  - Літній множник: {self.ga_tariff.seasonal_summer_multiplier:.3f}
  - Зимовий множник: {self.ga_tariff.seasonal_winter_multiplier:.3f}
  - Поріг знижки: {self.ga_tariff.volume_discount_threshold:.0f} кВт*год
  - Розмір знижки: {self.ga_tariff.volume_discount_rate:.1%}

Градієнтний спуск:
  - Базова ставка: {self.gd_tariff.base_rate:.3f} грн/кВт*год
  - Піковий множник: {self.gd_tariff.peak_multiplier:.3f}
  - Непіковий множник: {self.gd_tariff.off_peak_multiplier:.3f}
  - Літній множник: {self.gd_tariff.seasonal_summer_multiplier:.3f}
  - Зимовий множник: {self.gd_tariff.seasonal_winter_multiplier:.3f}
  - Поріг знижки: {self.gd_tariff.volume_discount_threshold:.0f} кВт*год
  - Розмір знижки: {self.gd_tariff.volume_discount_rate:.1%}

2. ФІНАНСОВІ РЕЗУЛЬТАТИ
-----------------------

Генетичний алгоритм:
  - Загальний дохід: {ga_metrics['total_revenue']:,.0f} грн
  - Дохід на кВт*год: {ga_metrics['revenue_per_kwh']:.3f} грн
  - Середній рахунок (побутові): {ga_metrics['avg_residential_bill']:,.0f} грн/рік
  - Середній рахунок (комерційні): {ga_metrics['avg_commercial_bill']:,.0f} грн/рік
  - Середній рахунок (промислові): {ga_metrics['avg_industrial_bill']:,.0f} грн/рік

Градієнтний спуск:
  - Загальний дохід: {gd_metrics['total_revenue']:,.0f} грн
  - Дохід на кВт*год: {gd_metrics['revenue_per_kwh']:.3f} грн
  - Середній рахунок (побутові): {gd_metrics['avg_residential_bill']:,.0f} грн/рік
  - Середній рахунок (комерційні): {gd_metrics['avg_commercial_bill']:,.0f} грн/рік
  - Середній рахунок (промислові): {gd_metrics['avg_industrial_bill']:,.0f} грн/рік

3. ЕФЕКТИВНІСТЬ ОПТИМІЗАЦІЇ
---------------------------

Генетичний алгоритм:
  - Значення цільової функції: {comparison['genetic_algorithm']['objective_value']:.6f}
  - Кількість поколінь: {comparison['genetic_algorithm']['convergence_time']}
  - Фінальна пристосованість: {comparison['genetic_algorithm']['final_fitness']:.6f}

Градієнтний спуск:
  - Значення цільової функції: {comparison['gradient_descent']['objective_value']:.6f}
  - Кількість ітерацій: {comparison['gradient_descent']['iterations']}

4. ПОРІВНЯЛЬНИЙ АНАЛІЗ
---------------------

Різниця в доходах: {abs(ga_metrics['total_revenue'] - gd_metrics['total_revenue']):,.0f} грн
Різниця в цільовій функції: {abs(comparison['genetic_algorithm']['objective_value'] - comparison['gradient_descent']['objective_value']):.6f}

Кращий метод за цільовою функцією: {'Генетичний алгоритм' if comparison['genetic_algorithm']['objective_value'] < comparison['gradient_descent']['objective_value'] else 'Градієнтний спуск'}
"""
        return report

def main():
    print("Оптимізація тарифів: Генетичний алгоритм vs Градієнтний спуск")
    print("=" * 70)
    
    print("\n1. СТВОРЕННЯ СИМУЛЯТОРА СПОЖИВАЧІВ")
    print("-" * 40)
    
    simulator = EnergyConsumerSimulator(n_consumers=200, n_days=90)
    print(f"Створено симулятор:")
    print(f"  - Кількість споживачів: {simulator.n_consumers}")
    print(f"  - Період симуляції: {simulator.n_days} днів")
    print(f"  - Загальних записів споживання: {len(simulator.consumption_data)}")
    
    type_stats = simulator.consumers_data.groupby('type').size()
    print(f"  - Розподіл споживачів:")
    for ctype, count in type_stats.items():
        print(f"    * {ctype}: {count} ({count/simulator.n_consumers:.1%})")
    
    print("\n2. ОПТИМІЗАЦІЯ ГЕНЕТИЧНИМ АЛГОРИТМОМ")
    print("-" * 40)
    
    ga_optimizer = GeneticAlgorithmOptimizer(
        simulator, 
        population_size=30,
        generations=50,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    ga_tariff, ga_history = ga_optimizer.optimize()
    
    print(f"\nРезультати генетичного алгоритму:")
    print(f"  - Базова ставка: {ga_tariff.base_rate:.3f} грн/кВт*год")
    print(f"  - Піковий множник: {ga_tariff.peak_multiplier:.3f}")
    print(f"  - Непіковий множник: {ga_tariff.off_peak_multiplier:.3f}")
    print(f"  - Літній множник: {ga_tariff.seasonal_summer_multiplier:.3f}")
    print(f"  - Зимовий множник: {ga_tariff.seasonal_winter_multiplier:.3f}")
    
    print("\n3. ОПТИМІЗАЦІЯ ГРАДІЄНТНИМ СПУСКОМ")
    print("-" * 40)
    
    gd_optimizer = GradientDescentOptimizer(simulator)
    gd_tariff, gd_history = gd_optimizer.optimize()
    
    print(f"\nРезультати градієнтного спуску:")
    print(f"  - Базова ставка: {gd_tariff.base_rate:.3f} грн/кВт*год")
    print(f"  - Піковий множник: {gd_tariff.peak_multiplier:.3f}")
    print(f"  - Непіковий множник: {gd_tariff.off_peak_multiplier:.3f}")
    print(f"  - Літній множник: {gd_tariff.seasonal_summer_multiplier:.3f}")
    print(f"  - Зимовий множник: {gd_tariff.seasonal_winter_multiplier:.3f}")
    
    
    print("\n4. ПОРІВНЯЛЬНИЙ АНАЛІЗ")
    print("-" * 40)
    
    analyzer = ResultsAnalyzer(ga_tariff, ga_history, gd_tariff, gd_history, simulator)
    comparison = analyzer.compare_tariffs()
    
    print(f"Порівняння ефективності:")
    print(f"  - ГА цільова функція: {comparison['genetic_algorithm']['objective_value']:.6f}")
    print(f"  - ГС цільова функція: {comparison['gradient_descent']['objective_value']:.6f}")
    
    better_method = "Генетичний алгоритм" if comparison['genetic_algorithm']['objective_value'] < comparison['gradient_descent']['objective_value'] else "Градієнтний спуск"
    print(f"  - Кращий метод: {better_method}")
    
    
    print("\n5. ФІНАНСОВИЙ АНАЛІЗ")
    print("-" * 40)
    
    ga_metrics = analyzer.calculate_financial_metrics(ga_tariff)
    gd_metrics = analyzer.calculate_financial_metrics(gd_tariff)
    
    print(f"Фінансові результати:")
    print(f"  ГА - Загальний дохід: {ga_metrics['total_revenue']:,.0f} грн")
    print(f"  ГС - Загальний дохід: {gd_metrics['total_revenue']:,.0f} грн")
    print(f"  ГА - Дохід/кВт*год: {ga_metrics['revenue_per_kwh']:.3f} грн")
    print(f"  ГС - Дохід/кВт*год: {gd_metrics['revenue_per_kwh']:.3f} грн")
    
    print("\n6. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
    print("-" * 40)
    
    print("Створення графіків порівняння...")
    
    analyzer.plot_convergence_comparison()
    
    analyzer.plot_tariff_comparison()
    
    analyzer.plot_consumer_impact_analysis()
    
    analyzer.plot_seasonal_analysis()
    
    
    print("\n7. ДЕТАЛЬНИЙ ЗВІТ")
    print("-" * 40)
    
    detailed_report = analyzer.generate_detailed_report()
    print(detailed_report)
    
    with open('tariff_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    print("\nДетальний звіт збережено у файл: tariff_optimization_report.txt")
    
    
    print("\n8. ПІДСУМОК ВИКОНАННЯ")
    print("=" * 50)
    
    print(f"\nРекомендований метод: {better_method}")
    return ga_tariff, gd_tariff, analyzer

if __name__ == "__main__":
    ga_result, gd_result, analysis = main()