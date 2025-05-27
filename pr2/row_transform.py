import pandas as pd
import numpy as np
import re
import time
import string
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} виконано за: {end_time - start_time:.4f} секунд")
        return result
    return wrapper

class TextTransformer:
    @staticmethod
    def clean_and_normalize(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Zа-яА-ЯіІїЇєЄ0-9\s]', '', text)
        
        return text.strip()
    
    @staticmethod
    def capitalize_words(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return text.title()
    
    @staticmethod
    def remove_digits(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def extract_emails(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return ', '.join(emails) if emails else ""
    
    @staticmethod
    def count_words(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text.split())

def generate_sample_data(n_rows=100000):
    print(f"Генерація {n_rows} рядків тестових даних...")
    
    sample_texts = [
        "  Hello WORLD!!!  This is a TEST   ",
        "john.doe@example.com contacted us yesterday",
        "Price: $29.99 for item #12345",
        "   UPPERCASE TEXT WITH    SPACES   ",
        "Mixed CaSe TeXt WiTh Numbers123 and symbols!@#",
        "Електронна пошта: test@ukr.net",
        "Текст українською мовою з цифрами 2024",
        "",
        None,
        "Multiple    spaces     between words",
    ]
    
    np.random.seed(42)
    data = {
        'id': range(1, n_rows + 1),
        'original_text': np.random.choice(sample_texts, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'length': np.random.randint(1, 100, n_rows)
    }
    
    return pd.DataFrame(data)

@measure_time
def transform_with_apply(df, column_name, transform_func):
    return df[column_name].apply(transform_func)

@measure_time
def transform_with_map(df, column_name, transform_func):
    return df[column_name].map(transform_func)

def compare_performance(df, column_name, transform_func):
    print(f"\nПорівняння продуктивності для {len(df)} рядків:")
    print("=" * 50)
    
    result_apply = transform_with_apply(df, column_name, transform_func)
    
    result_map = transform_with_map(df, column_name, transform_func)
    
    print(f"Результати ідентичні: {result_apply.equals(result_map)}")
    
    return result_apply

def demonstrate_text_transformations():
    df = generate_sample_data(50000)
    
    print("\nОригінальні дані (перші 10 рядків):")
    print("=" * 60)
    print(df[['id', 'original_text']].head(10))
    
    print(f"\n1. ОЧИЩЕННЯ ТА НОРМАЛІЗАЦІЯ")
    print("=" * 60)
    df['cleaned_text'] = compare_performance(df, 'original_text', TextTransformer.clean_and_normalize)
    
    print("\nРезультат очищення (перші 10 рядків):")
    comparison_df = df[['original_text', 'cleaned_text']].head(10)
    for idx, row in comparison_df.iterrows():
        print(f"Оригінал: '{row['original_text']}'")
        print(f"Очищено:  '{row['cleaned_text']}'")
        print("-" * 40)
    
    print(f"\n2. КАПІТАЛІЗАЦІЯ СЛІВ")
    print("=" * 60)
    df['capitalized_text'] = transform_with_apply(df, 'cleaned_text', TextTransformer.capitalize_words)
    
    print("\nРезультат капіталізації (перші 5 рядків):")
    for idx, row in df[['cleaned_text', 'capitalized_text']].head(5).iterrows():
        print(f"До:    '{row['cleaned_text']}'")
        print(f"Після: '{row['capitalized_text']}'")
        print("-" * 40)
    
    print(f"\n3. ВИДАЛЕННЯ ЦИФР")
    print("=" * 60)
    df['no_digits_text'] = transform_with_map(df, 'original_text', TextTransformer.remove_digits)
    
    print("\nРезультат видалення цифр (перші 5 рядків):")
    for idx, row in df[['original_text', 'no_digits_text']].head(5).iterrows():
        print(f"До:    '{row['original_text']}'")
        print(f"Після: '{row['no_digits_text']}'")
        print("-" * 40)
    
    print(f"\n4. ВИТЯГУВАННЯ EMAIL АДРЕС")
    print("=" * 60)
    df['extracted_emails'] = transform_with_apply(df, 'original_text', TextTransformer.extract_emails)
    
    email_results = df[df['extracted_emails'] != ''][['original_text', 'extracted_emails']].head()
    if not email_results.empty:
        print("\nЗнайдені email адреси:")
        for idx, row in email_results.iterrows():
            print(f"Текст: '{row['original_text']}'")
            print(f"Email: '{row['extracted_emails']}'")
            print("-" * 40)
    
    print(f"\n5. ПІДРАХУНОК СЛІВ")
    print("=" * 60)
    df['word_count'] = transform_with_apply(df, 'original_text', TextTransformer.count_words)
    
    print("\nСтатистика кількості слів:")
    print(df['word_count'].describe())
    
    return df

def analyze_memory_usage(df):
    print(f"\n6. АНАЛІЗ ВИКОРИСТАННЯ ПАМ'ЯТІ")
    print("=" * 60)
    
    memory_usage = df.memory_usage(deep=True)
    print("Використання пам'яті по колонках:")
    for col, usage in memory_usage.items():
        print(f"{col}: {usage / 1024 / 1024:.2f} MB")
    
    total_memory = memory_usage.sum() / 1024 / 1024
    print(f"\nЗагальне використання пам'яті: {total_memory:.2f} MB")

def demonstrate_advanced_transformations():
    print(f"\n7. СКЛАДНІ ТРАНСФОРМАЦІЇ")
    print("=" * 60)
    
    advanced_texts = [
        "Компанія ТОВ 'Техносервіс' надає послуги з 2018 року. Контакт: info@techservice.ua",
        "Замовлення №12345 на суму 1500.00 грн виконано 15.03.2024",
        "   УВАГА!!! Акція діє до 31.12.2024. Знижка 25%   ",
        "Студент Іван Петренко (група КП-21) склав іспит на 95 балів",
        "Email для зв'язку: support@university.edu.ua або admin@kpi.ua"
    ]
    
    df_advanced = pd.DataFrame({
        'id': range(1, len(advanced_texts) + 1),
        'text': advanced_texts
    })
    
    print("Оригінальні тексти:")
    for idx, text in enumerate(df_advanced['text'], 1):
        print(f"{idx}. {text}")
    
    def complex_transform(text):
        if pd.isna(text):
            return {"cleaned": "", "emails": "", "numbers": "", "words": 0}
        
        return {
            "cleaned": TextTransformer.clean_and_normalize(text),
            "emails": TextTransformer.extract_emails(text),
            "numbers": re.findall(r'\d+(?:\.\d+)?', str(text)),
            "words": TextTransformer.count_words(text)
        }
    
    results = df_advanced['text'].apply(complex_transform)
    
    df_advanced['cleaned_text'] = results.apply(lambda x: x['cleaned'])
    df_advanced['extracted_emails'] = results.apply(lambda x: x['emails'])
    df_advanced['extracted_numbers'] = results.apply(lambda x: ', '.join(map(str, x['numbers'])))
    df_advanced['word_count'] = results.apply(lambda x: x['words'])
    
    print(f"\nРезультати комплексної трансформації:")
    print("=" * 60)
    for idx, row in df_advanced.iterrows():
        print(f"ID: {row['id']}")
        print(f"Оригінал: {row['text']}")
        print(f"Очищено: {row['cleaned_text']}")
        print(f"Email: {row['extracted_emails']}")
        print(f"Числа: {row['extracted_numbers']}")
        print(f"Слова: {row['word_count']}")
        print("-" * 60)
    
    return df_advanced

def main():
    print("Трансформація рядків за допомогою .apply() та .map()")
    print("=" * 80)
    
    df_main = demonstrate_text_transformations()
    
    analyze_memory_usage(df_main)
    
    df_advanced = demonstrate_advanced_transformations()
    
    print(f"\nОброблено {len(df_main)} основних записів та {len(df_advanced)} складних прикладів")
    
    return df_main, df_advanced

if __name__ == "__main__":
    df_main, df_advanced = main()
    
    print(f"\nДОДАТКОВА ІНФОРМАЦІЯ:")
    print("=" * 40)
    print(f"Розмір основного DataFrame: {df_main.shape}")
    print(f"Колонки: {list(df_main.columns)}")
    print(f"Типи даних:")
    print(df_main.dtypes)