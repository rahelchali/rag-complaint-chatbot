import pandas as pd
import numpy as np
import os
from src.eda_preprocessing import CrediTrustDataPipeline
from sentence_transformers import SentenceTransformer

def execute_complete_ingestion_sprint():
    print("[!] Activating CrediTrust Financial Automated Data Engineering Pipeline...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    
    np.random.seed(42)
    size = 100
    mock_products = ["Credit card or prepaid card", "Consumer Loan", "Checking or savings account", "Money transfer, virtual currency, or money service"]
    
    mock_raw = pd.DataFrame({
        "Complaint ID": [f"ID_{100000+i}" for i in range(size)],
        "Product": np.random.choice(mock_products, size=size, p=[0.4, 0.2, 0.2, 0.2]),
        "Consumer complaint narrative": ["Bank charged unauthorized fees. I am writing to file a complaint because my card was declined during transactions." if i % 3 != 0 else "" for i in range(size)]
    })
    
    pipeline = CrediTrustDataPipeline(chunk_size=500, chunk_overlap=50)
    cleaned_df = pipeline.execute_eda_and_cleaning(mock_raw)
    cleaned_df.to_csv("data/filtered_complaints.csv", index=False)
    
    stratified_sample = pipeline.generate_stratified_sample(cleaned_df, target_sample_size=30)
    chunks_df = pipeline.build_local_vector_chunks(stratified_sample)
    
    print("[!] Loading embedding model candidates: sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"[!] Transforming {len(chunks_df)} text chunks into 384-dimensional vector embeddings...")
    embeddings = model.encode(chunks_df["text_chunk"].tolist(), show_progress_bar=True)
    
    chunks_df["vector_embedding"] = list(embeddings)
    chunks_df.to_parquet("vector_store/chroma_index.parquet", index=False)
    chunks_df.to_csv("data/processed/complaint_chunks.csv", index=False)
    
    print("\n[✓] VECTOR STORE PERSISTED ARCHITECTURE INDEX GENERATED:")
    print(f"   [✓] Embedding Array Dimensions : {embeddings.shape} Vectors Matrix")
    print(f"   [✓] Database Format Target    : Persisted Parquet Vector Index")
    print(f"   [✓] Local Storage Location    : vector_store/chroma_index.parquet")
    print(f"   [✓] Total Chunks Indexed      : {len(chunks_df)} blocks recorded.")

if __name__ == "__main__":
    execute_complete_ingestion_sprint()
