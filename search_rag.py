"""
Оптимизированный поиск с сохранением исправленных запросов
"""

import os
import pandas as pd
from tqdm import tqdm
import re
import hashlib
import json
import random
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"
INPUT_Q = "data/questions_clean.csv"
OUTPUT = "RAG_results_v4.csv"
OUTPUT_CLEANED_QUERIES = "data_clean/questions_cleaned.csv"  # Новый файл с очищенными запросами
CACHE_FILE = "query_cache.json"


class SmartQueryProcessor:
    def __init__(self):
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cache(self):
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")

    def smart_clean_query(self, query: str) -> str:
        """Умная очистка запроса с сохранением оригинального"""
        if not isinstance(query, str):
            return ""

        original_query = query

        # Кэширование
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        if query_hash in self.cache:
            return self.cache[query_hash], original_query

        # Базовая нормализация
        cleaned = query.lower().strip()

        # Удаление мусорных символов
        cleaned = re.sub(r'[^\w\s\-\?\!\.,]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Семантические замены на основе паттернов
        patterns = [
            (r'\bне\s+прих\w*', 'не приходит'),
            (r'\bсмс\w*', 'смс'),
            (r'\bсчёт\w*', 'счет'),
            (r'\bр\/с\b', 'расчетный счет'),
            (r'\bбик\b', 'БИК'),
            (r'\bальфа\w*', 'Альфа-Банк'),
            (r'\bкред\w*\s+карт\w*', 'кредитная карта'),
            (r'\bномер\s+счет\w*', 'номер счета'),
        ]

        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned)

        # Восстановление капитализации для читабельности
        cleaned = cleaned.capitalize()

        self.cache[query_hash] = cleaned
        return cleaned, original_query


class GuaranteedSearch:
    def __init__(self, db):
        self.db = db
        self.all_web_ids = self._get_all_web_ids()

    def _get_all_web_ids(self) -> List[int]:
        """Получить все доступные web_id из базы"""
        try:
            all_docs = self.db.get()
            web_ids = []
            for meta in all_docs.get("metadatas", []):
                if meta and "web_id" in meta and meta["web_id"] not in web_ids:
                    web_ids.append(meta["web_id"])
            return web_ids
        except Exception as e:
            print(f"Ошибка получения web_ids: {e}")
            return list(range(1, 100))

    def search_with_fallback(self, query: str, k: int = 5) -> List[int]:
        """Поиск с гарантированным возвратом k результатов"""
        try:
            # Основной поиск
            docs = self.db.similarity_search(query, k=k * 2)
            found_web_ids = []

            for doc in docs:
                web_id = doc.metadata.get("web_id")
                if web_id is not None and web_id not in found_web_ids:
                    found_web_ids.append(web_id)
                if len(found_web_ids) >= k:
                    break

            # Если нашли достаточно - возвращаем
            if len(found_web_ids) >= k:
                return found_web_ids[:k]

            # Дополняем до k случайными уникальными web_id
            remaining_slots = k - len(found_web_ids)
            if remaining_slots > 0:
                available_ids = [id for id in self.all_web_ids if id not in found_web_ids]
                if available_ids:
                    additional_ids = random.sample(
                        available_ids,
                        min(remaining_slots, len(available_ids))
                    )
                    found_web_ids.extend(additional_ids)

                # Если все еще не хватает - дополняем существующими
                while len(found_web_ids) < k:
                    found_web_ids.append(found_web_ids[0] if found_web_ids else self.all_web_ids[0])

            return found_web_ids[:k]

        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return random.sample(self.all_web_ids, min(k, len(self.all_web_ids)))


def main():
    print("Загрузка Chroma DB...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        print("База данных загружена")
    except Exception as e:
        print(f"Ошибка загрузки базы данных: {e}")
        return

    # Инициализация поиска с гарантиями
    searcher = GuaranteedSearch(db)
    processor = SmartQueryProcessor()

    print("Загрузка вопросов...")
    qdf = pd.read_csv(INPUT_Q)
    print(f"Всего вопросов: {len(qdf)}")

    # Очистка запросов с сохранением оригиналов
    print("Очистка запросов...")
    cleaned_queries_data = []

    for q in tqdm(qdf["query"].fillna(""), desc="Очистка"):
        cleaned_query, original_query = processor.smart_clean_query(q)
        cleaned_queries_data.append({
            'original_query': original_query,
            'cleaned_query': cleaned_query
        })

    # Сохраняем отдельный файл с очищенными запросами
    cleaned_queries_df = pd.DataFrame(cleaned_queries_data)
    cleaned_queries_df.to_csv(OUTPUT_CLEANED_QUERIES, index=False, encoding='utf-8')
    print(f"Очищенные запросы сохранены в {OUTPUT_CLEANED_QUERIES}")

    # Поиск с гарантированным результатом
    print("Поиск релевантных документов...")
    results = []

    for _, row in tqdm(qdf.iterrows(), total=len(qdf), desc="Поиск"):
        q_id = row["q_id"]
        original_query = row.get("original_query", "").strip()
        cleaned_query = row.get("query_cleaned", "").strip()

        # Используем очищенный запрос, или оригинальный если очищенный пустой
        search_query = cleaned_query if cleaned_query else original_query

        # Гарантированный поиск 5 документов
        web_list = searcher.search_with_fallback(search_query, k=5)

        # Дополнительная проверка - гарантируем ровно 5 элементов
        while len(web_list) < 5:
            web_list.append(web_list[0] if web_list else searcher.all_web_ids[0])

        # Сохраняем только q_id и web_list для итогового файла
        results.append({
            "q_id": q_id,
            "web_list": web_list
        })

    # Сохранение результатов (только q_id и web_list)
    output_df = pd.DataFrame(results)

    # Проверяем что все web_list имеют ровно 5 элементов
    for i, row in output_df.iterrows():
        if len(row['web_list']) != 5:
            print(f"Внимание: q_id {row['q_id']} имеет {len(row['web_list'])} элементов")

    output_df.to_csv(OUTPUT, index=False)

    # Сохранение кэша
    processor.save_cache()

    print(f"Результаты сохранены в {OUTPUT}")
    print(f"Обработано запросов: {len(results)}")
    print(f"Уникальных web_id в базе: {len(searcher.all_web_ids)}")


if __name__ == "__main__":
    main()