"""
Оптимизированное создание Chroma DB с сохранением исправленного текста
"""

import os
import pandas as pd
from tqdm import tqdm
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"
INPUT_CSV = "data/websites_updated.csv"
OUTPUT_CLEANED_CSV = "data_clean/websites_cleaned.csv"  # Новый файл с очищенными текстами
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_WORKERS = 2


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

        return text if len(text) > 10 else ""


class AdvancedTextCleaner:
    def __init__(self, use_nn=True):
        self.normalizer = FastTextNormalizer() if use_nn else None
        self.common_abbreviations = {
            'т.д.': 'так далее', 'т.п.': 'тому подобное', 'т.е.': 'то есть',
            'т.к.': 'так как', 'др.': 'другие', 'пр.': 'прочие',
            'см.': 'смотри', 'напр.': 'например', 'стр.': 'страница',
        }

    def clean_text_advanced(self, text):
        """Гибридная очистка: быстрая нейросеть + правила"""
        if not isinstance(text, str) or not text.strip():
            return ""

        original_text = text

        # Используем нейросеть для сложных случаев
        if self.normalizer and len(text) > 50 and len(text) < 1000:
            cleaned = self.normalizer.normalize_with_nn(text)
        else:
            cleaned = self.normalizer.fallback_clean(text) if self.normalizer else self.fast_clean(text)

        # Дополнительная rule-based очистка
        cleaned = self.fast_clean(cleaned)

        # Возвращаем оригинал и очищенный текст
        result = cleaned if len(cleaned) > 15 else ""
        return result, original_text

    def fast_clean(self, text):
        """Быстрая rule-based очистка"""
        text = re.sub(r'\s+', ' ', text).strip()

        for abbr, full in self.common_abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)

        return text


def process_document_row(row, cleaner, splitter):
    """Обработка одной строки документа с сохранением оригинального текста"""
    text = str(row.get("text", "") or "")
    title = str(row.get("title", "") or "")

    # Очистка текста с сохранением оригинала
    cleaned_text, original_text = cleaner.clean_text_advanced(text)
    cleaned_title, original_title = cleaner.clean_text_advanced(title)

    if not cleaned_text:
        return []

    # Подготовка данных для сохранения
    row_data = row.to_dict()
    row_data['original_text'] = original_text
    row_data['cleaned_text'] = cleaned_text
    row_data['original_title'] = original_title
    row_data['cleaned_title'] = cleaned_title

    # Разбиение на чанки если нужно
    if len(cleaned_text) > CHUNK_SIZE:
        try:
            chunks = splitter.split_text(cleaned_text)
            return [(chunk, cleaned_title, row_data) for chunk in chunks if chunk.strip()]
        except Exception:
            return [(cleaned_text, cleaned_title, row_data)]
    else:
        return [(cleaned_text, cleaned_title, row_data)]


def main():
    print("Загрузка CSV...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено {len(df)} страниц")

    # Инициализация компонентов
    cleaner = AdvancedTextCleaner(use_nn=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # Параллельная обработка
    print("Обработка документов...")
    all_docs = []
    all_cleaned_data = []  # Для сохранения очищенных данных

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(process_document_row, row, cleaner, splitter)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка"):
            try:
                result = future.result()
                all_docs.extend(result)
                # Сохраняем данные для CSV (берем первый чанк как представитель)
                if result:
                    all_cleaned_data.append(result[0][2])  # row_data из первого чанка
            except Exception as e:
                print(f"Ошибка обработки: {e}")

    # Сохранение очищенных данных в CSV
    print("Сохранение очищенных данных...")
    if all_cleaned_data:
        cleaned_df = pd.DataFrame(all_cleaned_data)
        # Сохраняем только нужные колонки
        output_columns = ['web_id', 'url', 'kind', 'original_title', 'cleaned_title', 'original_text', 'cleaned_text']
        available_columns = [col for col in output_columns if col in cleaned_df.columns]
        cleaned_df[available_columns].to_csv(OUTPUT_CLEANED_CSV, index=False, encoding='utf-8')
        print(f"Очищенные данные сохранены в {OUTPUT_CLEANED_CSV}")

    # Подготовка документов для Chroma
    print("Подготовка документов для векторной БД...")
    chroma_docs = []
    doc_ids = []

    for i, (content, title, row_data) in enumerate(all_docs):
        if not content:
            continue

        doc_id = hashlib.md5(f"{row_data['web_id']}_{i}".encode()).hexdigest()

        chroma_docs.append(
            Document(
                page_content=content,
                metadata={
                    "web_id": row_data.get("web_id"),
                    "title": title,
                    "url": row_data.get("url"),
                    "kind": row_data.get("kind"),
                    "chunk_id": i,
                }
            )
        )
        doc_ids.append(doc_id)

    print(f"Создано {len(chroma_docs)} чанков")

    if not chroma_docs:
        print("Нет документов для создания базы")
        return

    # Создание векторной базы
    print("Создание Chroma DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma.from_documents(
        documents=chroma_docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        ids=doc_ids
    )
    db.persist()

    print(f"Chroma DB создана в {DB_DIR}")
    print(f"Статистика: {len(df)} -> {len(chroma_docs)} чанков")
    print(f"Очищенные данные: {OUTPUT_CLEANED_CSV}")


if __name__ == "__main__":
    main()