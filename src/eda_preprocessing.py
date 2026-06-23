import pandas as pd
import numpy as np
import re

class CrediTrustDataPipeline:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def execute_eda_and_cleaning(self, raw_df):
        """Processes raw complaints, flags narrative completeness, and normalizes text streams."""
        df = raw_df.copy()
        product_map = {
            'Credit card or prepaid card': 'Credit Cards',
            'Credit card': 'Credit Cards',
            'Consumer Loan': 'Personal Loans',
            'Checking or savings account': 'Savings Accounts',
            'Money transfer, virtual currency, or money service': 'Money Transfers'
        }
        df['Standardized_Product'] = df['Product'].map(product_map).fillna('Other')

        has_narrative_mask = df['Consumer complaint narrative'].notnull() & (df['Consumer complaint narrative'].str.strip() != '')
        df = df[df['Standardized_Product'] != 'Other']
        df = df[has_narrative_mask].copy()

        def clean_text(text):
            if not isinstance(text, str): return ""
            text = text.lower()
            text = re.sub(r'i am writing to file a complaint|xxxx|xx/xx/\d{4}', '', text)
            text = re.sub(r'[^a-z0-9\s\.,!\?]', '', text)
            return re.sub(r'\s+', ' ', text).strip()

        df['Cleaned_Narrative'] = df['Consumer complaint narrative'].apply(clean_text)
        df['Narrative_Word_Count'] = df['Cleaned_Narrative'].apply(lambda x: len(x.split()))
        return df

    def generate_stratified_sample(self, cleaned_df, target_sample_size=1500):
        """Executes a proportional stratified sample layer across product cohorts."""
        groups = cleaned_df.groupby('Standardized_Product')
        sampled_chunks = []
        for name, group in groups:
            proportion = len(group) / len(cleaned_df)
            group_sample_size = int(np.round(proportion * target_sample_size))
            group_sample_size = max(10, min(group_sample_size, len(group)))
            sampled_chunks.append(group.sample(n=min(group_sample_size, len(group)), replace=True, random_state=42) if len(group) > 0 else pd.DataFrame())
        return pd.concat(sampled_chunks).reset_index(drop=True)

    def build_local_vector_chunks(self, stratified_df):
        """Splits narratives into fixed character sizes with sliding overlap parameters."""
        chunked_records = []
        for _, row in stratified_df.iterrows():
            text = row['Cleaned_Narrative']
            complaint_id = row.get('Complaint ID', 'UNKNOWN_ID')
            category = row['Standardized_Product']
            start = 0
            text_len = len(text)
            idx = 0
            while start < text_len:
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunked_records.append({
                    "chunk_id": f"CHUNK_{complaint_id}_{idx}",
                    "complaint_id": complaint_id,
                    "product_category": category,
                    "text_chunk": chunk_text
                })
                start += (self.chunk_size - self.chunk_overlap)
                idx += 1
        return pd.DataFrame(chunked_records)