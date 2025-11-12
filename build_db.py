# build_db.py

import pandas as pd
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"

# === Функция очистки текста ===
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # привести к нижнему регистру
    text = re.sub(r"<.*?>", " ", text)  # удалить HTML-теги
    text = re.sub(r"\s+", " ", text)  # заменить несколько пробелов одним
    text = re.sub(r"[^\w\s.,!?-]", " ", text)  # удалить спецсимволы, кроме пунктуации
    text = text.strip()  # удалить пробелы в начале и конце
    return text

# === Шаг 1. Загрузка данных ===
websites_df = pd.read_csv("data/websites_updated.csv")
print(f"Загружено {len(websites_df)} веб-страниц")

# === Шаг 2. Очистка данных ===
# Заменяем NaN на пустую строку
websites_df["text"] = websites_df["text"].fillna("")

# Очистка текста
websites_df["text"] = websites_df["text"].apply(clean_text)

# Удаляем полностью пустые строки
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
