# build_db.py

import re
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"

# === Функция очистки текста ===
def clean_text_rag(text: str) -> str:
    """
    Оптимизированная очистка текста для RAG-систем.
    Удаляет спецсимволы, эмодзи, разметку и нормализует пробелы.
    """
    if not isinstance(text, str):
        return ""

    # Привести к нижнему регистру
    text = text.lower()

    # Удалить URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Удалить email-адреса
    text = re.sub(r'\S*@\S*\s?', ' ', text)

    # Удалить управляющие символы и неразрывные пробелы
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = text.replace('\xa0', ' ')

    # Удалить эмодзи и специальные символы (расширенный набор)
    text = re.sub(r'[^\w\s.,:;!?()\-/а-яa-z0-9]', ' ', text)

    # Удалить маркеры списков и буллиты
    text = re.sub(r'[\u2022\u25CF\uF0A7•◆◇▶️▪️▫️★☆✔️✳️❖\-\*\+]+', ' ', text)

    # Удалить HTML-теги
    text = re.sub(r'<.*?>', ' ', text)

    # Удалить повторяющиеся пунктуации
    text = re.sub(r'([.,:;!?])\1+', r'\1', text)

    # Нормализовать дефисы
    text = re.sub(r'[–—]', '-', text)

    # Удалить всё, кроме букв, цифр, основных знаков пунктуации и пробелов
    text = re.sub(r'[^а-яa-z0-9\s.,:;!?()\-]', ' ', text)

    # Удалить отдельно стоящие дефисы
    text = re.sub(r'\s-\s', ' ', text)

    # Удалить повторяющиеся пробелы
    text = re.sub(r'\s+', ' ', text)

    # Удалить пробелы вокруг пунктуации
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    text = re.sub(r'([.,:;!?])\s+', r'\1 ', text)

    # Удалить ведущие и конечные пробелы
    text = text.strip()

    return text

# Дополнительная функция для семантической очистки
def semantic_clean(text: str) -> str:
    """
    Семантическая очистка - удаление малозначащих фрагментов
    """
    # Удалить короткие неинформативные строки
    if len(text) < 20:
        return ""

    # Удалить строки, состоящие в основном из цифр/пунктуации
    if re.match(r'^[\d\s\W]+$', text):
        return ""

    return text

# Комплексная функция обработки для RAG
def process_text_for_rag(text: str) -> str:
    """
    Полная обработка текста для RAG-систем
    """
    # Очистка текста
    cleaned = clean_text_rag(text)

    # Семантическая фильтрация
    cleaned = semantic_clean(cleaned)

    return cleaned


# === Шаг 1. Загрузка данных ===
websites_df = pd.read_csv("data/websites_updated.csv")
print(f"Загружено {len(websites_df)} веб-страниц")

# === Шаг 2. Очистка данных ===
websites_df["text"] = websites_df["text"].fillna("").apply(process_text_for_rag)

# Убираем полностью пустые тексты
websites_df = websites_df[websites_df["text"].str.strip() != ""]

print(f"После очистки осталось {len(websites_df)} страниц с текстом")

# === Шаг 3. Подготовка документов ===
docs = [
    Document(
        page_content=row["text"],
        metadata={
            "web_id": row["web_id"],
            "title": row["title"],
            "url": row["url"],
            "kind": row["kind"],
        },
    )
    for _, row in websites_df.iterrows()
]

# === Шаг 4. Создание эмбеддингов ===
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# === Шаг 5. Создание и сохранение базы Chroma ===
print("Создание ChromaDB...")
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=DB_DIR
)

db.persist()
print(f"✅ База знаний успешно создана и сохранена в папке '{DB_DIR}'")
