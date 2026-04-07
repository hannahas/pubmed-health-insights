import requests
import pandas as pd
import time

def fetch_pubmed_ids(query, max_results=200):
    """Search PubMed and return a list of article IDs matching the query."""
    """Uses free public API from NCBI (PubMed) to query PubMed"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["esearchresult"]["idlist"]


def fetch_abstract(pmid):
    """Fetch the abstract and metadata for a single PubMed article by ID."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    return response.text


def parse_abstract(xml_text, pmid):
    """Extract title, abstract, and year from raw PubMed XML."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
        title = root.findtext(".//ArticleTitle") or ""
        abstract = root.findtext(".//AbstractText") or ""
        year = root.findtext(".//PubDate/Year") or ""
        return {"pmid": pmid, "title": title, "abstract": abstract, "year": year}
    except Exception as e:
        print(f"Error parsing PMID {pmid}: {e}")
        return None


def fetch_all_abstracts(query, max_results=200):
    """Fetch all abstracts for a query and return as a DataFrame."""
    print(f"Searching PubMed for: {query}")
    pmids = fetch_pubmed_ids(query, max_results)
    print(f"Found {len(pmids)} articles. Fetching abstracts...")

    records = []
    for i, pmid in enumerate(pmids):
        xml_text = fetch_abstract(pmid)
        record = parse_abstract(xml_text, pmid)
        if record:
            records.append(record)
        time.sleep(0.34)  # stay within NCBI rate limit of 3 requests/sec
        if (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1} of {len(pmids)}...")

    df = pd.DataFrame(records)
    print(f"Done. Retrieved {len(df)} abstracts.")
    return df


if __name__ == "__main__":
    query = "TCR repertoire sequencing"
    df = fetch_all_abstracts(query, max_results=200)
    df.to_csv("data/abstracts.csv", index=False)
    print("Saved to data/abstracts.csv")