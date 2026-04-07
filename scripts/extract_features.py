import anthropic
import pandas as pd
import json
import time
from dotenv import load_dotenv

load_dotenv()

def extract_features(abstract, title):
    """Use Claude to extract structured features from a single abstract."""
    client = anthropic.Anthropic()

    prompt = f"""You are a biomedical research analyst. Extract structured information from the following research abstract and return ONLY a JSON object with no other text.

Title: {title}
Abstract: {abstract}

Return a JSON object with exactly these fields:
{{
    "study_type": "clinical" or "basic_research" or "computational" or "review",
    "sample_size": the number of human subjects or samples if mentioned, otherwise null,
    "technology": the sequencing or analysis technology used (e.g. "10x Genomics", "bulk TCR-seq", "scRNA-seq"), or null,
    "disease_focus": the primary disease or condition studied, or "none" if not disease-focused,
    "key_finding": a single sentence summarizing the main finding,
    "clinical_relevance": "high", "medium", or "low"
}}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    # Strip markdown code blocks if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    return json.loads(response_text.strip())


def extract_all_features(input_path, output_path):
    """Process all abstracts and save extracted features."""
    df = pd.read_csv(input_path)
    print(f"Processing {len(df)} abstracts...")

    results = []
    for i, row in df.iterrows():
        try:
            features = extract_features(row['abstract'], row['title'])
            features['pmid'] = row['pmid']
            features['title'] = row['title']
            features['year'] = row['year']
            results.append(features)
        except Exception as e:
            print(f"Error on PMID {row['pmid']}: {e}")
            results.append({'pmid': row['pmid'], 'error': str(e)})

        time.sleep(0.5)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} of {len(df)}...")

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")
    return output_df


if __name__ == "__main__":
    extract_all_features(
        input_path="data/abstracts.csv",
        output_path="data/extracted_features.csv"
    )