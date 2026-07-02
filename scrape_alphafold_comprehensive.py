#!/usr/bin/env python3
"""
Comprehensive scrape of AlphaFold small-molecule drug discovery papers from PubMed.
Detects clinical trial / preclinical status from abstracts.
Output: alphafold_comprehensive.csv
"""

import csv
import re
import time
import xml.etree.ElementTree as ET
import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

QUERY = (
    'alphafold[tiab] AND '
    '('
        '"small molecule"[tiab] OR inhibitor[tiab] OR '
        '"structure-based drug"[tiab] OR "virtual screening"[tiab] OR '
        'docking[tiab] OR "lead compound"[tiab] OR "hit compound"[tiab] OR '
        'IC50[tiab] OR ADMET[tiab] OR pharmacophore[tiab] OR '
        '"lead optimization"[tiab] OR "drug design"[tiab] OR '
        '"drug discovery"[tiab] OR "binding affinity"[tiab] OR '
        'scaffold[tiab] OR "active site"[tiab]'
    ') '
    'NOT Review[pt] NOT "Systematic Review"[pt] NOT "Meta-Analysis"[pt] '
    'NOT (vaccine[ti] OR antibody[ti] OR "gene therapy"[ti] OR "mRNA vaccine"[ti])'
)

# --- Status detection patterns ---

CLINICAL_PATTERNS = [
    (r'phase\s*(I{1,3}|[123])\s*(\/|or)?\s*(II{1,2}|[23])?\s*clinical\s*trial', 'Clinical Trial'),
    (r'phase\s*[123I]{1,3}\b', 'Clinical Trial'),
    (r'clinical\s*trial', 'Clinical Trial'),
    (r'first[\s-]in[\s-]human', 'Clinical Trial'),
    (r'fda[\s-]?approved', 'Approved Drug'),
    (r'approved\s*(drug|therapy|treatment)', 'Approved Drug'),
    (r'entered\s*clinical', 'Clinical Trial'),
    (r'clinical\s*candidate', 'Clinical Candidate'),
    (r'ind\s*application', 'Clinical Candidate'),
    (r'investigational\s*new\s*drug', 'Clinical Candidate'),
]

PRECLINICAL_PATTERNS = [
    (r'in\s*vivo\s*(efficacy|study|studies|model|data|experiment)', 'Preclinical (In Vivo)'),
    (r'(mouse|rat|murine|rodent|xenograft|tumor\s*model)\s*(model|study|studies|experiment)', 'Preclinical (In Vivo)'),
    (r'animal\s*(model|study|studies|experiment)', 'Preclinical (In Vivo)'),
    (r'pharmacokinetic', 'Preclinical (PK/PD)'),
    (r'\bpk(/pd)?\b.*\b(study|data|profile|parameter)', 'Preclinical (PK/PD)'),
    (r'in\s*vivo', 'Preclinical (In Vivo)'),
    (r'(oral\s*bioavailability|bioavailability)', 'Preclinical (PK/PD)'),
    (r'(tumor|cancer|xenograft)\s*(growth|regression|inhibition)', 'Preclinical (In Vivo)'),
]


def detect_status(abstract: str) -> str:
    if not abstract:
        return "In Vitro / Computational"
    text = abstract.lower()
    for pattern, label in CLINICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    for pattern, label in PRECLINICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return "In Vitro / Computational"


def search_pubmed(query, retmax=2000):
    url = f"{BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = data["esearchresult"].get("idlist", [])
    count = int(data["esearchresult"].get("count", 0))
    return ids, count


def fetch_details_batch(pmids, batch_size=200):
    all_papers = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        url = f"{BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        all_papers.extend(parse_xml(r.text))
        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} records...", flush=True)
        time.sleep(0.4)
    return all_papers


def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        paper = {}

        title_el = article.find(".//ArticleTitle")
        paper["Title"] = "".join(title_el.itertext()).strip() if title_el is not None else ""

        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())
        paper["Authors"] = "; ".join(authors) if authors else ""

        journal_el = article.find(".//Journal/Title")
        paper["Journal"] = journal_el.text.strip() if journal_el is not None else ""

        pub_date = article.find(".//PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4]
        paper["Year"] = year

        pmid_el = article.find(".//PMID")
        paper["PMID"] = pmid_el.text.strip() if pmid_el is not None else ""

        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text.strip()
                break
        paper["DOI"] = doi
        paper["PubMed URL"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['PMID']}/" if paper["PMID"] else ""

        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        paper["Abstract"] = " ".join(abstract_parts)

        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
        paper["Publication Type"] = "; ".join(pub_types)

        keywords = [kw.text.strip() for kw in article.findall(".//Keyword") if kw.text]
        mesh = [m.findtext("DescriptorName", "").strip() for m in article.findall(".//MeshHeading")]
        paper["Keywords"] = "; ".join(keywords + [m for m in mesh if m])

        paper["Drug Discovery Status"] = detect_status(paper["Abstract"])

        papers.append(paper)
    return papers


def main():
    print(f"Searching PubMed...")
    pmids, count = search_pubmed(QUERY)
    print(f"Found {count} papers total, fetching up to {len(pmids)}...")

    papers = fetch_details_batch(pmids)
    papers.sort(key=lambda x: x.get("Year", "0"), reverse=True)

    # Summary
    from collections import Counter
    status_counts = Counter(p["Drug Discovery Status"] for p in papers)
    print(f"\nStatus breakdown:")
    for status, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {n}")

    out_file = "/Users/joyal/lab-website-2/alphafold_comprehensive.csv"
    fieldnames = [
        "Title", "Authors", "Journal", "Year",
        "Drug Discovery Status", "PMID", "DOI", "PubMed URL",
        "Publication Type", "Keywords", "Abstract"
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)

    print(f"\nDone! {len(papers)} papers saved to {out_file}")


if __name__ == "__main__":
    main()
