# scripts/1_build_knowledge_base.py

import pandas as pd
import numpy as np
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from dotenv import load_dotenv


# ======================================================================================
# 1. HELPER FUNCTIONS (Developed during our data exploration)
# ======================================================================================

def robust_clean_html(data):
    """
    Handles list or string input, cleans HTML tags, and returns clean text.
    """
    text_to_clean = ""
    if isinstance(data, list) and len(data) > 0:
        text_to_clean = data[0]
    elif isinstance(data, str):
        text_to_clean = data

    if not isinstance(text_to_clean, str):
        return ""  # Return empty string if input is not text

    soup = BeautifulSoup(text_to_clean, "lxml")
    return soup.get_text(separator=" ", strip=True)


def is_empty_definitive(value):
    """
    Definitive function that correctly handles all known empty cases,
    including the literal string 'None'.
    """
    if value == 'None':
        return True
    if isinstance(value, (list, dict, np.ndarray)):
        return len(value) == 0
    if pd.isnull(value):
        return True
    if isinstance(value, str):
        return value.strip() == ''
    return False


# ======================================================================================
# 2. MAIN ETL SCRIPT
# ======================================================================================

def create_knowledge_document(product: dict) -> str:
    """
    Creates a single text document for a product, handling missing fields.
    """
    # Start with the most reliable field
    doc = f"Product Title: {product.get('title', 'N/A')}\n"

    # Add other fields only if they are not empty
    if not is_empty_definitive(product.get('store')):
        doc += f"Brand: {product.get('store')}\n"

    if not is_empty_definitive(product.get('price')):
        # Ensure price is formatted correctly
        try:
            price = float(product.get('price'))
            doc += f"Price: ${price:.2f}\n"
        except (ValueError, TypeError):
            pass  # Ignore if price is not a valid float

    if not is_empty_definitive(product.get('features')):
        cleaned_features = [robust_clean_html(f) for f in product.get('features')]
        if cleaned_features:
            doc += "Features:\n"
            for feature in cleaned_features:
                doc += f"- {feature}\n"

    if not is_empty_definitive(product.get('description')):
        cleaned_description = robust_clean_html(product.get('description'))
        if cleaned_description:
            doc += f"Description: {cleaned_description}\n"

    return doc


def main():
    """
    Main function to run the ETL pipeline.
    """
    # --- Load Environment Variables ---
    # This will load the HF_HOME variable from the .env file in the project root
    load_dotenv()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        print(f"Using Hugging Face cache directory: {hf_home}")
    else:
        print("HF_HOME not set. Using default Hugging Face cache directory.")

    print("\nStarting Step 1.1: Data Processing Pipeline...")

    # --- EXTRACT ---
    print("Loading raw electronics metadata from Hugging Face...")
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics", trust_remote_code=True)
    meta_df = pd.DataFrame(dataset['full'])
    print(f"Loaded {len(meta_df)} product records.")

    # --- TRANSFORM ---
    print("Transforming data and generating knowledge documents...")
    knowledge_base = []

    # Using tqdm for a progress bar
    for _, row in tqdm(meta_df.iterrows(), total=meta_df.shape[0], desc="Processing products"):
        product_dict = row.to_dict()

        # We only need the product ID and the final text document
        knowledge_document = {
            "parent_asin": product_dict.get('parent_asin'),
            "knowledge_doc": create_knowledge_document(product_dict)
        }
        knowledge_base.append(knowledge_document)

    # --- LOAD ---
    print("Saving the generated knowledge base to a file...")
    knowledge_df = pd.DataFrame(knowledge_base)

    # The path now correctly points from 'scripts/' up to the root, then into 'data/'
    output_path = "../data/knowledge_base.csv"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    knowledge_df.to_csv(output_path, index=False)
    print(f"Successfully created knowledge base at: {output_path}")
    print(f"Total documents created: {len(knowledge_df)}")


if __name__ == "__main__":
    main()
