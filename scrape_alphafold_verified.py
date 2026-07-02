#!/usr/bin/env python3
"""
Verified scrape: research papers where AlphaFold was actually used as a tool
in small-molecule drug discovery. Mines review papers for additional leads.
Output: alphafold_verified.csv
"""

import csv
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── Searches ──────────────────────────────────────────────────────────────────

PRIMARY_QUERY = (
    '(alphafold[tiab] OR "alphafold2"[tiab] OR "alpha fold"[tiab]) '
    'AND ('
        '"virtual screening"[tiab] OR "molecular docking"[tiab] OR '
        '"structure-based drug"[tiab] OR "structure-based design"[tiab] OR '
        '"drug design"[tiab] OR "drug discovery"[tiab] OR '
        '"hit identification"[tiab] OR "lead optimization"[tiab] OR '
        '"lead identification"[tiab] OR "compound screening"[tiab]'
    ') '
    'AND ('
        'inhibitor[tiab] OR "small molecule"[tiab] OR '
        'IC50[tiab] OR Ki[tiab] OR EC50[tiab] OR '
        '"binding affinity"[tiab] OR scaffold[tiab] OR '
        '"active compound"[tiab] OR "hit compound"[tiab]'
    ') '
    'NOT Review[pt] NOT "Systematic Review"[pt] NOT "Meta-Analysis"[pt] '
    'NOT (vaccine[ti] OR antibody[ti] OR "gene therapy"[ti] OR siRNA[ti])'
)

REVIEW_QUERY = (
    '(alphafold[tiab] OR "alphafold2"[tiab]) AND '
    '"drug discovery"[tiab] AND '
    '(Review[pt] OR "Systematic Review"[pt])'
)

# ── AlphaFold usage scoring ────────────────────────────────────────────────────

def score_alphafold_usage(title: str, abstract: str) -> tuple[int, str]:
    """
    Score how confident we are that AlphaFold was actually used as a tool,
    not just cited as background. Returns (score, reason).
    Score >= 3 → include. Score < 3 → exclude.
    """
    text = (title + " " + abstract).lower()
    score = 0
    reasons = []

    # Strong evidence: AlphaFold near structural/methodological terms
    if re.search(r'alphafold.{0,40}(structure|model|predict)', text):
        score += 2; reasons.append("AF used for structure prediction")
    if re.search(r'(structure|model|predict).{0,40}alphafold', text):
        score += 1; reasons.append("structure from AF")
    if re.search(r'alphafold.{0,60}(dock|virtual screen|binding site|active site)', text):
        score += 2; reasons.append("AF structure used for docking/screening")
    if re.search(r'(dock|virtual screen|binding site).{0,60}alphafold', text):
        score += 1; reasons.append("docking on AF structure")

    # Explicit usage statements
    if re.search(r'(using|used|employ|utiliz|obtain|generat).{0,20}alphafold', text):
        score += 2; reasons.append("explicit AF usage stated")
    if re.search(r'alphafold.{0,20}(was used|were used|enabled|facilitat|allow)', text):
        score += 2; reasons.append("explicit AF usage stated")

    # AF2/AF3 variants
    if re.search(r'\baf2\b|\baf3\b', text):
        score += 1; reasons.append("AF2/AF3 mentioned")
    if re.search(r'alphafold[\s-]?(2|3|multimer)', text):
        score += 1; reasons.append("AF2/AF3/Multimer mentioned")

    # AlphaFold DB / deposited structures
    if re.search(r'alphafold\s*(database|db|repository|model\s*db)', text):
        score += 1; reasons.append("AF database structure used")

    # Activity data present (confirms experimental drug discovery)
    has_activity = bool(re.search(
        r'\b(IC50|EC50|Ki\b|Kd\b|GI50|CC50|MIC\b|pIC50|pEC50|'
        r'percent\s*inhibit|%\s*inhibit|\d+\s*[nmuμ][Mm]\b)', text))
    if has_activity:
        score += 1; reasons.append("activity data reported")

    # Synthesis / experimental validation
    if re.search(r'(synthesi[sz]|synthesized|prepared|characteriz)', text):
        score += 1; reasons.append("synthesis reported")

    # Penalise if AlphaFold only in intro/background context
    if re.search(r'(background|introduction|previously\s*described|'
                 r'has\s*been\s*shown|it\s*has\s*been\s*demonstrated).{0,80}alphafold', text):
        score -= 1; reasons.append("AF possibly background-only")

    return score, "; ".join(reasons)


def detect_status(abstract: str) -> str:
    if not abstract:
        return "Computational Only"
    t = abstract.lower()
    if re.search(r'(phase\s*[123i]{1,3}|clinical\s*trial|first.in.human|'
                 r'fda.approved|ema.approved|approved\s*drug)', t):
        return "Clinical / Approved"
    if re.search(r'(clinical\s*candidate|ind\s*application|'
                 r'investigational\s*new\s*drug|entered\s*clinic)', t):
        return "Clinical Candidate"
    if re.search(r'(in\s*vivo|animal\s*model|mouse\s*model|rat\s*model|'
                 r'xenograft|pharmacokinetic|oral\s*bioavailability|'
                 r'tumor\s*(growth|regression)|murine)', t):
        return "Preclinical (In Vivo)"
    if re.search(r'(cell\s*(viability|proliferation|based\s*assay)|'
                 r'cellular\s*assay|cytotoxic|antiproliferat|'
                 r'western\s*blot|flow\s*cytometry)', t):
        return "In Vitro (Cell-Based)"
    if re.search(r'(IC50|EC50|Ki\b|Kd\b|MIC\b|pIC50|inhibit|bind)', t):
        return "In Vitro (Biochemical)"
    return "Computational Only"


def extract_target(title: str, abstract: str) -> str:
    text = title + " " + abstract
    patterns = [
        (r'\b(EGFR|HER2|KRAS|BRAF|PARP|CDK[0-9]+|BCL-?2|BCL-?XL|MDM2|'
         r'p53|VEGFR|PDGFR|ALK|ROS1|MET|AKT|mTOR|PI3K|JAK[0-9]*|STAT[0-9]*|'
         r'BTK|SRC|ABL|FLT3|IDH[12]|DNMT|HDAC|BRD[24])\b', None),
        (r'\b(ACE2|ACE|TMPRSS2|3CL.pro|Mpro|RdRp|neuraminidase|'
         r'helicase|protease|integrase|reverse\s*transcriptase)\b', None),
        (r'\b(ROCK[12]?|Rho\s*kinase|DYRK1A|GSK-?3|CK1|PLK1|'
         r'Aurora\s*[AB]|CHK[12]|WEE1|ATM|ATR)\b', None),
        (r'\b(COX-?[12]|cyclooxygenase|lipoxygenase|LOX|5-LOX|'
         r'phosphodiesterase|PDE[0-9]*|adenosine\s*receptor|'
         r'dopamine\s*receptor|serotonin\s*receptor|GPCR)\b', None),
        (r'\b(TAAR[0-9]*|D[12345]R|5-HT[0-9A-Z]+|'
         r'mGluR[0-9]*|NMDA|AMPA|GABA[AB]?)\b', None),
        (r'\b(dihydrofolate\s*reductase|DHFR|thymidylate\s*synthase|'
         r'topoisomerase|gyrase|DNA\s*polymerase|RNA\s*polymerase)\b', None),
    ]
    for pattern, _ in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    # fallback: look for "X inhibitor" or "inhibitor of X"
    m = re.search(r'([A-Z][A-Za-z0-9\-]{2,20})\s+inhibitor', text)
    if m:
        return m.group(1)
    return ""


# ── PubMed API helpers ─────────────────────────────────────────────────────────

def search_pubmed(query, retmax=2000):
    r = requests.get(f"{BASE_URL}/esearch.fcgi", params={
        "db": "pubmed", "term": query,
        "retmax": retmax, "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["esearchresult"].get("idlist", []), int(data["esearchresult"].get("count", 0))


def get_references(pmid):
    """Get PMIDs of papers cited by a given PMID via elink."""
    r = requests.get(f"{BASE_URL}/elink.fcgi", params={
        "dbfrom": "pubmed", "linkname": "pubmed_pubmed_refs",
        "id": pmid, "retmode": "json",
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    refs = []
    for linkset in data.get("linksets", []):
        for linksetdb in linkset.get("linksetdbs", []):
            refs.extend(linksetdb.get("links", []))
    return refs


def fetch_batch(pmids, batch_size=200):
    all_papers = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        r = requests.get(f"{BASE_URL}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(batch),
            "rettype": "xml", "retmode": "xml",
        }, timeout=60)
        r.raise_for_status()
        all_papers.extend(parse_xml(r.text))
        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} records", flush=True)
        time.sleep(0.35)
    return all_papers


def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        p = {}
        title_el = article.find(".//ArticleTitle")
        p["Title"] = "".join(title_el.itertext()).strip() if title_el is not None else ""

        authors = []
        for a in article.findall(".//Author"):
            last = a.findtext("LastName", "")
            fore = a.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())
        p["Authors"] = "; ".join(authors)

        jnl = article.find(".//Journal/Title")
        p["Journal"] = jnl.text.strip() if jnl is not None else ""

        pub_date = article.find(".//PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4]
        p["Year"] = year

        pmid_el = article.find(".//PMID")
        p["PMID"] = pmid_el.text.strip() if pmid_el is not None else ""

        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text.strip(); break
        p["DOI"] = doi
        p["PubMed URL"] = f"https://pubmed.ncbi.nlm.nih.gov/{p['PMID']}/" if p["PMID"] else ""

        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        p["Abstract"] = " ".join(abstract_parts)

        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
        p["Publication Type"] = "; ".join(pub_types)

        kws = [kw.text.strip() for kw in article.findall(".//Keyword") if kw.text]
        mesh = [m.findtext("DescriptorName", "").strip() for m in article.findall(".//MeshHeading")]
        p["Keywords/MeSH"] = "; ".join(kws + [m for m in mesh if m])

        papers.append(p)
    return papers


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seen = set()
    all_pmids = []

    # 1. Primary search
    print("Step 1: Primary PubMed search...")
    pmids, count = search_pubmed(PRIMARY_QUERY)
    print(f"  Found {count} papers, retrieved {len(pmids)}")
    for p in pmids:
        if p not in seen:
            seen.add(p); all_pmids.append(p)

    # 2. Mine review papers for cited research papers
    print("\nStep 2: Mining review papers for cited research papers...")
    review_pmids, rcount = search_pubmed(REVIEW_QUERY, retmax=50)
    print(f"  Found {rcount} review papers, mining top {len(review_pmids)}")
    ref_count = 0
    for i, rpid in enumerate(review_pmids[:30]):
        try:
            refs = get_references(rpid)
            for ref in refs:
                if ref not in seen:
                    seen.add(ref); all_pmids.append(ref)
                    ref_count += 1
            time.sleep(0.35)
        except Exception as e:
            pass
    print(f"  Added {ref_count} unique cited papers from reviews")

    # 3. Fetch all paper details
    print(f"\nStep 3: Fetching details for {len(all_pmids)} total candidates...")
    papers = fetch_batch(all_pmids)

    # 4. Filter: keep only non-review research papers with genuine AlphaFold use
    print(f"\nStep 4: Filtering {len(papers)} papers...")
    filtered = []
    excluded_review = 0
    excluded_low_score = 0

    for p in papers:
        # Drop reviews
        pt = p.get("Publication Type", "").lower()
        if any(x in pt for x in ["review", "systematic review", "meta-analysis"]):
            excluded_review += 1
            continue

        # Score AlphaFold usage
        score, reason = score_alphafold_usage(p["Title"], p["Abstract"])
        if score < 3:
            excluded_low_score += 1
            continue

        # Annotate
        p["AlphaFold Use Score"] = score
        p["AlphaFold Use Evidence"] = reason
        p["Drug Discovery Status"] = detect_status(p["Abstract"])
        p["Primary Target"] = extract_target(p["Title"], p["Abstract"])
        filtered.append(p)

    print(f"  Kept: {len(filtered)} | Removed reviews: {excluded_review} | Low AF score: {excluded_low_score}")

    # 5. Deduplicate by PMID
    seen_pmids = set()
    deduped = []
    for p in filtered:
        if p["PMID"] not in seen_pmids:
            seen_pmids.add(p["PMID"]); deduped.append(p)
    print(f"  After dedup: {len(deduped)} papers")

    # 6. Sort by year desc, then by status
    status_order = {
        "Clinical / Approved": 0, "Clinical Candidate": 1,
        "Preclinical (In Vivo)": 2, "In Vitro (Cell-Based)": 3,
        "In Vitro (Biochemical)": 4, "Computational Only": 5,
    }
    deduped.sort(key=lambda x: (
        -int(x.get("Year", "0") or 0),
        status_order.get(x.get("Drug Discovery Status", ""), 9)
    ))

    # Summary
    print("\nStatus breakdown:")
    for status, n in sorted(Counter(p["Drug Discovery Status"] for p in deduped).items(),
                             key=lambda x: status_order.get(x[0], 9)):
        print(f"  {status}: {n}")

    # 7. Write CSV
    out_file = "/Users/joyal/lab-website-2/alphafold_verified.csv"
    fieldnames = [
        "Title", "Authors", "Journal", "Year",
        "Drug Discovery Status", "Primary Target",
        "AlphaFold Use Score", "AlphaFold Use Evidence",
        "PMID", "DOI", "PubMed URL",
        "Publication Type", "Keywords/MeSH", "Abstract"
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nDone! {len(deduped)} verified papers → {out_file}")


if __name__ == "__main__":
    main()
