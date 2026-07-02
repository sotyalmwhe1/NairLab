#!/usr/bin/env python3
"""
Scrape AlphaFold drug discovery papers from PubMed across target journals.
Outputs: alphafold_drug_discovery_papers.csv
"""

import csv
import time
import xml.etree.ElementTree as ET
import requests

JOURNALS = [
    "Drug Discov Today",
    "J Med Chem",
    "Eur J Med Chem",
    "ACS Med Chem Lett",
    "Bioorg Med Chem Lett",
    "Bioorg Med Chem",
    "J Comput Aided Mol Des",
    "Curr Opin Drug Discov Devel",
    "Expert Opin Drug Discov",
    "Drug Dev Res",
    "SLAS Discov",
    "Nat Rev Drug Discov",
    "Drug Des Devel Ther",
    "Mol Pharm",
    "ChemMedChem",
    "Future Med Chem",
    "Br J Pharmacol",
    "Pharm Res",
    "Eur J Pharm Sci",
    "J Pharmacol Exp Ther",
]

JOURNAL_DISPLAY = {
    "Drug Discov Today": "Drug Discovery Today",
    "J Med Chem": "Journal of Medicinal Chemistry",
    "Eur J Med Chem": "European Journal of Medicinal Chemistry",
    "ACS Med Chem Lett": "ACS Medicinal Chemistry Letters",
    "Bioorg Med Chem Lett": "Bioorganic & Medicinal Chemistry Letters",
    "Bioorg Med Chem": "Bioorganic & Medicinal Chemistry",
    "J Comput Aided Mol Des": "Journal of Computer-Aided Molecular Design",
    "Curr Opin Drug Discov Devel": "Current Opinion in Drug Discovery & Development",
    "Expert Opin Drug Discov": "Expert Opinion on Drug Discovery",
    "Drug Dev Res": "Drug Discovery and Development",
    "SLAS Discov": "SLAS Discovery",
    "Nat Rev Drug Discov": "Nature Reviews Drug Discovery",
    "Drug Des Devel Ther": "Drug Design, Development and Therapy",
    "Mol Pharm": "Molecular Pharmaceutics",
    "ChemMedChem": "ChemMedChem",
    "Future Med Chem": "Future Medicinal Chemistry",
    "Br J Pharmacol": "British Journal of Pharmacology",
    "Pharm Res": "Pharmaceutical Research",
    "Eur J Pharm Sci": "European Journal of Pharmaceutical Sciences",
    "J Pharmacol Exp Ther": "Journal of Pharmacology and Experimental Therapeutics",
}

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query, retmax=500):
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


def fetch_details(pmids):
    if not pmids:
        return []
    url = f"{BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return parse_xml(r.text)


def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        paper = {}

        # Title
        title_el = article.find(".//ArticleTitle")
        paper["Title"] = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())
        paper["Authors"] = "; ".join(authors) if authors else ""

        # Journal
        journal_el = article.find(".//Journal/Title")
        paper["Journal"] = journal_el.text.strip() if journal_el is not None else ""

        # Year
        pub_date = article.find(".//PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4]
        paper["Year"] = year

        # PMID
        pmid_el = article.find(".//PMID")
        paper["PMID"] = pmid_el.text.strip() if pmid_el is not None else ""

        # DOI
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text.strip()
                break
        paper["DOI"] = doi
        paper["URL"] = f"https://doi.org/{doi}" if doi else (
            f"https://pubmed.ncbi.nlm.nih.gov/{paper['PMID']}/" if paper["PMID"] else ""
        )

        # Abstract
        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        paper["Abstract"] = " ".join(abstract_parts)

        # Keywords
        keywords = [kw.text.strip() for kw in article.findall(".//Keyword") if kw.text]
        paper["Keywords"] = "; ".join(keywords)

        papers.append(paper)
    return papers


SMALL_MOL_FILTER = (
    '("small molecule"[tiab] OR inhibitor[tiab] OR compound[tiab] OR scaffold[tiab] '
    'OR ligand[tiab] OR "lead compound"[tiab] OR "hit compound"[tiab] OR '
    '"drug-like"[tiab] OR docking[tiab] OR "virtual screening"[tiab] '
    'OR "binding affinity"[tiab] OR IC50[tiab] OR ADMET[tiab] OR pharmacophore[tiab] '
    'OR QSAR[tiab] OR "lead optimization"[tiab])'
)

EXCLUDE_BIOLOGICS = (
    'NOT (vaccine[ti] OR antibody[ti] OR "gene therapy"[ti] OR "mRNA vaccine"[ti])'
)


def main():
    all_papers = []
    seen_pmids = set()

    for journal_abbr in JOURNALS:
        display = JOURNAL_DISPLAY.get(journal_abbr, journal_abbr)
        query = (
            f'alphafold[tiab] AND "{journal_abbr}"[Journal] '
            f'AND {SMALL_MOL_FILTER} '
            f'{EXCLUDE_BIOLOGICS} '
            f'NOT Review[pt] NOT "Systematic Review"[pt] NOT "Meta-Analysis"[pt]'
        )
        print(f"Searching: {display} ...", end=" ", flush=True)

        try:
            pmids, count = search_pubmed(query)
            print(f"{count} found", end="")

            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            if new_pmids:
                time.sleep(0.4)
                papers = fetch_details(new_pmids)
                all_papers.extend(papers)
                print(f" → fetched {len(papers)}")
            else:
                print()
        except Exception as e:
            print(f" ERROR: {e}")

        time.sleep(0.4)

    # Sort by year descending
    all_papers.sort(key=lambda x: x.get("Year", "0"), reverse=True)

    out_file = "/Users/joyal/lab-website-2/alphafold_drug_discovery_papers.csv"
    fieldnames = ["Title", "Authors", "Journal", "Year", "PMID", "DOI", "URL", "Keywords", "Abstract"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_papers)

    print(f"\nDone! {len(all_papers)} papers saved to {out_file}")


if __name__ == "__main__":
    main()
