import os
import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Настройки ===
INPUT_CSV = "../data/websites_updated.csv"
OUTPUT_CLEANED_CSV = "../data_clean/websites_cleaned.csv"
MAX_WORKERS = 4  # Количество потоков для обработки


# === Быстрая нейросетевая модель для нормализации ===
class FastTextNormalizer:
    def __init__(self):
        try:
            self.model_name = "cointegrated/rut5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Модель нормализации загружена на {self.device}")
        except Exception as e:
            print(f"Нейросетевая нормализация недоступна: {e}")
            self.model = None

    def normalize_with_nn(self, text, max_length=512):
        """Нормализация текста с помощью нейросети"""
        if not self.model or not text.strip():
            return self.fallback_clean(text)

        try:
            input_text = "нормализуй текст и раскрой сокращения: " + text[:max_length]

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result if result.strip() else self.fallback_clean(text)

        except Exception as e:
            return self.fallback_clean(text)

    def fallback_clean(self, text):
        """Резервная очистка регулярками"""
        if not isinstance(text, str):
            return ""

        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S*@\S*\s?', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^\w\s\.,:;!?\-()/а-яА-Яa-zA-Z0-9]', ' ', text)
        text = re.sub(r'[\u2022\u25CF\uf0a7•◆◇▶▪▫★☆✔✳❖\-\*\+]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class AdvancedTextCleaner:
    def __init__(self, use_nn=True):
        self.normalizer = FastTextNormalizer() if use_nn else None
        self.common_abbreviations = {
            'т.д.': 'так далее', 'т.п.': 'тому подобное', 'т.е.': 'то есть',
            'т.к.': 'так как', 'др.': 'другие', 'пр.': 'прочие',
            'см.': 'смотри', 'напр.': 'например', 'стр.': 'страница',
        }

    def clean_text_advanced(self, text, is_title=False):
        """Гибридная очистка: быстрая нейросеть + правила"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Для title убираем ограничение по длине
        if is_title:
            # Используем нейросеть для title независимо от длины
            if self.normalizer:
                cleaned = self.normalizer.normalize_with_nn(text)
            else:
                cleaned = self.fast_clean(text)
        else:
            # Для обычного текста сохраняем ограничения
            if self.normalizer and len(text) > 50 and len(text) < 1000:
                cleaned = self.normalizer.normalize_with_nn(text)
            else:
                cleaned = self.normalizer.fallback_clean(text) if self.normalizer else self.fast_clean(text)

        # Дополнительная rule-based очистка
        cleaned = self.fast_clean(cleaned)

        # Для title возвращаем любой результат, для текста - только если длиннее 15 символов
        return cleaned if (is_title or len(cleaned) > 15) else ""

    def fast_clean(self, text):
        """Быстрая rule-based очистка"""
        text = re.sub(r'\s+', ' ', text).strip()

        for abbr, full in self.common_abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)

        return text


def process_document_row(row, cleaner):
    """Обработка одной строки документа"""
    text = str(row.get("text", "") or "")
    title = str(row.get("title", "") or "")

    # Очистка текста (для title снимаем ограничения по длине)
    cleaned_text = cleaner.clean_text_advanced(text, is_title=False)
    cleaned_title = cleaner.clean_text_advanced(title, is_title=True)

    # Подготовка данных для сохранения
    row_data = row.to_dict()
    row_data['cleaned_text'] = cleaned_text
    row_data['cleaned_title'] = cleaned_title

    return row_data


def main():
    print("Загрузка CSV...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено {len(df)} страниц")

    # Инициализация очистителя
    cleaner = AdvancedTextCleaner(use_nn=True)

    # Параллельная обработка
    print("Обработка документов...")
    all_cleaned_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(process_document_row, row, cleaner)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка"):
            try:
                result = future.result()
                all_cleaned_data.append(result)
            except Exception as e:
                print(f"Ошибка обработки: {e}")

    # Сохранение очищенных данных в CSV
    print("Сохранение очищенных данных...")
    if all_cleaned_data:
        cleaned_df = pd.DataFrame(all_cleaned_data)

        # Сохраняем только нужные колонки
        output_columns = ['web_id', 'url', 'kind', 'cleaned_title', 'cleaned_text']

        # Выбираем только существующие колонки
        available_columns = [col for col in output_columns if col in cleaned_df.columns]

        # Добавляем остальные колонки из исходных данных (кроме оригинальных текстов)
        for col in df.columns:
            if col not in available_columns and col in cleaned_df.columns and col not in ['text', 'title']:
                available_columns.append(col)

        cleaned_df[available_columns].to_csv(OUTPUT_CLEANED_CSV, index=False, encoding='utf-8')
        print(f"Очищенные данные сохранены в {OUTPUT_CLEANED_CSV}")
        print(f"Обработано записей: {len(all_cleaned_data)}")
    else:
        print("Нет данных для сохранения")


if __name__ == "__main__":
    # Создаем папку для результатов если её нет
    os.makedirs(os.path.dirname(OUTPUT_CLEANED_CSV), exist_ok=True)
    main()