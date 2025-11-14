import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from tqdm import tqdm

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "alfa_rag_documents"

# === Инициализация клиентов ===
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
embedding_model = SentenceTransformer(MODEL_NAME)


# === Очистка и пересоздание коллекции ===
def recreate_collection():
    """Удаляет старую коллекцию и создает новую"""
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Коллекция {COLLECTION_NAME} удалена")
    except Exception as e:
        print(f"Коллекция не существовала: {e}")

    # Создаем новую коллекцию
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )
    print(f"Коллекция {COLLECTION_NAME} создана")


# === Подготовка документов с cleaned_title в контексте ===
def prepare_documents(df: pd.DataFrame) -> list:
    """Создает документы с объединенным cleaned_title и cleaned_text"""
    documents = []

    # Проверяем наличие нужных колонок
    print(f"Колонки в файле: {list(df.columns)}")

    for _, row in df.iterrows():
        # Объединяем cleaned_title и cleaned_text для контекста
        full_content = f"{row['cleaned_title']} {row['cleaned_text']}".strip()

        document = {
            'id': str(uuid.uuid4()),
            'content': full_content,
            'metadata': {
                'web_id': row['web_id'],
                'title': row['cleaned_title'],
                'source': 'cleaned_data'
            }
        }
        documents.append(document)

    print(f"Подготовлено {len(documents)} документов")
    return documents


# === Разделение текста на чанки ===
def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Разделяет документы на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for doc in tqdm(documents, desc="Разделение документов на чанки"):
        splits = text_splitter.split_text(doc['content'])
        for i, split in enumerate(splits):
            chunk = {
                'id': f"{doc['id']}_chunk_{i}",
                'content': split,
                'metadata': {
                    **doc['metadata'],
                    'chunk_id': i,
                    'total_chunks': len(splits)
                }
            }
            chunks.append(chunk)

    print(f"Разделено на {len(chunks)} чанков")
    return chunks



# === Создание эмбеддингов и загрузка в Qdrant  ===
def upload_to_qdrant_detailed(chunks: list):
    """Создает эмбеддинги и загружает в Qdrant с детальным прогресс-баром"""
    points = []

    # Создаем эмбеддинги для всех чанков с детальным прогресс-баром
    contents = [chunk['content'] for chunk in chunks]
    print(f"Создание эмбеддингов для {len(contents)} чанков...")

    # Прогресс-бар для эмбеддингов
    embeddings = []
    batch_size = 32

    with tqdm(total=len(contents), desc="Эмбеддинги", unit="chunk") as pbar:
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            batch_embeddings = embedding_model.encode(
                batch_contents,
                show_progress_bar=False  # Отключаем встроенный прогресс-бар
            ).tolist()
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch_contents))

    print("Эмбеддинги созданы")

    # Создаем точки с прогресс-баром
    with tqdm(total=len(chunks), desc="Подготовка точек", unit="point") as pbar:
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=chunk['id'],
                vector=embeddings[i],
                payload={
                    'content': chunk['content'],
                    'web_id': chunk['metadata']['web_id'],
                    'title': chunk['metadata']['title'],
                    'chunk_id': chunk['metadata']['chunk_id'],
                    'total_chunks': chunk['metadata']['total_chunks'],
                    'source': chunk['metadata']['source']
                }
            )
            points.append(point)
            pbar.update(1)

    # Загружаем в Qdrant батчами с прогресс-баром
    with tqdm(total=len(points), desc="Загрузка в Qdrant", unit="batch") as pbar:
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            pbar.update(len(batch))

    print(f"Всего загружено {len(points)} векторов в Qdrant")


# === Основная функция ===
def main():
    # Загружаем очищенные данные
    input_csv = "../data_clean/websites_cleaned.csv"
    df = pd.read_csv(input_csv)
    print(f"Загружено {len(df)} записей из {input_csv}")

    # Очищаем и создаем коллекцию
    recreate_collection()

    # Подготавливаем документы
    documents = prepare_documents(df)

    # Разделяем на чанки
    chunks = split_documents(documents)

    # Загружаем в Qdrant
    upload_to_qdrant_detailed(chunks)

    # Проверяем результат
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"База данных готова! Векторов в коллекции: {collection_info.vectors_count}")


if __name__ == "__main__":
    main()