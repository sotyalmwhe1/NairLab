#!/usr/bin/env python3
"""
Semantic Scholar-based search for AlphaFold small molecule drug discovery papers.
Uses semantic search + citation network expansion from known seed papers.
Output: alphafold_semantic.csv
"""

import csv
import re
import time
import requests
from collections import Counter

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REC  = "https://api.semanticscholar.org/recommendations/v1"

FIELDS = (
    "title,authors,year,abstract,externalIds,journal,"
    "publicationTypes,referenceCount,citationCount,isOpenAccess"
)

# ── Known gold-standard papers (from Undermind / manually verified) ────────────
# These are used as seeds for citation-network expansion and recommendations.
SEED_TITLES = [
    "AlphaFold accelerated discovery of psychotropic agonists targeting the trace amine-associated receptor 1",
    "AlphaFold2 structures guide prospective ligand discovery",
    "AlphaFold accelerates artificial intelligence powered drug discovery: efficient discovery of a novel CDK20 small molecule inhibitor",
    "Discovery of novel and selective SIK2 inhibitors by the application of AlphaFold structures and generative models",
    "Leveraging Machine Learning and AlphaFold2 Steering to Discover State-Specific Inhibitors Across the Kinome",
    "Discovery of Covalent Ligands with AlphaFold3",
    "Discovery of Novel Chemotype LRRK2 Inhibitors Through AlphaFold2-Generated Structure-Based Docking Screen",
    "Utilization of an optimized AlphaFold protein model for structure-based design of a selective HDAC11 inhibitor",
    "Comparative Structure-Based Virtual Screening Utilizing Optimized AlphaFold Model Identifies Selective HDAC11 Inhibitor",
    "Deep contrastive learning enables genome-wide virtual screening",
    "AlphaFold3 and RoseTTAFold All-Atom structures enable radiosensitizers discovery",
    "Discovery of a cryptic pocket in the AI-predicted structure of PPM1D phosphatase",
    "AlphaFold Kinase Optimizer: Enhancing Virtual Screening Performance Through Automated Refinement",
]

# ── Semantic search queries ────────────────────────────────────────────────────
QUERIES = [
    "AlphaFold small molecule drug discovery hit identification",
    "AlphaFold2 prospective virtual screening ligand discovery",
    "AlphaFold structure-based drug design inhibitor synthesis",
    "AlphaFold docking hit compound identification kinase GPCR",
    "AlphaFold predicted structure virtual screening active compound",
    "AlphaFold2 structure-based design inhibitor biological activity",
    "AlphaFold drug target no crystal structure novel inhibitor",
]

# ── Status detection ───────────────────────────────────────────────────────────
def detect_stage(abstract: str) -> str:
    if not abstract:
        return "Stage unclear"
    t = abstract.lower()
    if re.search(r'(phase\s*[123i]{1,3}|clinical\s*trial|first.in.human|'
                 r'fda.approved|ema.approved|approved\s*drug|entered\s*clinic)', t):
        return "Clinical / Approved"
    if re.search(r'(clinical\s*candidate|ind\s*application|investigational\s*new\s*drug)', t):
        return "IND-enabling / Clinical Candidate"
    if re.search(r'(in\s*vivo|animal\s*model|mouse\s*model|rat\s*model|xenograft|'
                 r'pharmacokinetic|oral\s*bioavailability|tumor\s*(growth|regression)|murine|'
                 r'efficacy\s*in\s*(mice|rats|vivo))', t):
        return "In Vivo Preclinical"
    if re.search(r'(lead\s*optim|sar\s*(study|analys|exploration)|structure.activity|'
                 r'second.round|analog|scaffold\s*hop)', t):
        return "Lead Optimisation"
    if re.search(r'(hit\s*(compound|identif|to.lead)|synthesiz|IC50|EC50|Ki\b|Kd\b|'
                 r'cell\s*(viability|proliferation|based)|cellular\s*activ|antiproliferat)', t):
        return "Hit Identification / In Vitro"
    if re.search(r'(virtual\s*screen|dock|binding\s*affin|active\s*site|'
                 r'binding\s*site|hit\s*rate)', t):
        return "Computational / Hit Discovery"
    return "Stage unclear"


def is_review(pub_types):
    if not pub_types:
        return False
    return any(t.lower() in ("review", "editorialcomment") for t in pub_types)


def passes_filter(paper: dict) -> tuple[bool, str]:
    """Return (keep, reason). Only keep original research papers on AF + small mol drug discovery."""
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text = title + " " + abstract

    # Drop reviews
    if is_review(paper.get("publicationTypes") or []):
        return False, "review"

    # Must mention alphafold/af2/af3 explicitly
    if not re.search(r'\b(alphafold|alpha.fold|af2|af3)\b', text, re.IGNORECASE):
        return False, "no alphafold mention"

    # Must have evidence of actual use (not just cited in intro)
    af_used = bool(re.search(
        r'(alphafold.{0,50}(struct|model|predict|generat|database|db)|'
        r'(struct|model|predict).{0,50}alphafold|'
        r'(using|used|employ|utiliz|obtain).{0,30}alphafold|'
        r'alphafold.{0,30}(was used|were used|enabled|facilitat)|'
        r'\baf2\b.{0,50}(struct|model|dock|screen)|'
        r'\baf3\b.{0,50}(struct|model|dock|screen))', text, re.IGNORECASE))
    if not af_used:
        return False, "alphafold only mentioned in background"

    # Must have small molecule drug discovery context
    sm_hit = bool(re.search(
        r'(small.molecule|inhibitor|docking|virtual.scree|hit.identif|'
        r'drug.design|drug.discovery|lead.optim|IC50|EC50|\bKi\b|\bKd\b|'
        r'binding.affin|structure.based.drug|compound.screen|'
        r'novel.compound|active.compound|hit.compound)', text, re.IGNORECASE))
    if not sm_hit:
        return False, "no small molecule drug discovery context"

    # Exclude if primarily biologics/non-small-molecule
    biologic = bool(re.search(
        r'(^|\s)(vaccine|antibody|monoclonal|siRNA|CRISPR|gene.therapy|'
        r'mRNA.vaccine|cell.therapy|protein.engineering)(\s|$)', title, re.IGNORECASE))
    if biologic:
        return False, "biologic focus"

    return True, "pass"


# ── API helpers ────────────────────────────────────────────────────────────────
def s2_search(query: str, limit: int = 100) -> list[dict]:
    results = []
    offset = 0
    while offset < min(limit, 400):
        r = requests.get(f"{S2_BASE}/paper/search", params={
            "query": query,
            "fields": FIELDS,
            "limit": min(100, limit - offset),
            "offset": offset,
        }, timeout=30)
        if r.status_code == 429:
            time.sleep(5); continue
        r.raise_for_status()
        data = r.json()
        batch = data.get("data", [])
        results.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        time.sleep(1)
    return results


def s2_get_paper(title: str):
    r = requests.get(f"{S2_BASE}/paper/search", params={
        "query": title, "fields": FIELDS, "limit": 3,
    }, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    return data[0] if data else None


def s2_citations(paper_id: str, limit: int = 100) -> list[dict]:
    results = []
    offset = 0
    while offset < limit:
        r = requests.get(f"{S2_BASE}/paper/{paper_id}/citations", params={
            "fields": FIELDS, "limit": min(100, limit - offset), "offset": offset,
        }, timeout=30)
        if r.status_code == 429:
            time.sleep(5); continue
        if r.status_code != 200:
            break
        batch = r.json().get("data", [])
        results.extend(p["citingPaper"] for p in batch if "citingPaper" in p)
        if len(batch) < 100:
            break
        offset += 100
        time.sleep(1)
    return results


def s2_recommendations(paper_ids: list[str]) -> list[dict]:
    if not paper_ids:
        return []
    r = requests.post(f"{S2_REC}/papers/", json={
        "positivePaperIds": paper_ids[:20],
        "negativePaperIds": [],
    }, params={"fields": FIELDS, "limit": 100}, timeout=30)
    if r.status_code != 200:
        return []
    return r.json().get("recommendedPapers", [])


def fmt_paper(p: dict) -> dict:
    authors = p.get("authors") or []
    author_str = "; ".join(a.get("name", "") for a in authors[:10])
    journal = ""
    j = p.get("journal")
    if j:
        journal = j.get("name", "") or ""
    ext = p.get("externalIds") or {}
    doi = ext.get("DOI", "")
    pmid = ext.get("PubMed", "")
    arxiv = ext.get("ArXiv", "")
    url = (f"https://doi.org/{doi}" if doi else
           f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else
           f"https://arxiv.org/abs/{arxiv}" if arxiv else "")
    pub_types = p.get("publicationTypes") or []
    return {
        "Title": p.get("title", ""),
        "Authors": author_str,
        "Journal": journal,
        "Year": str(p.get("year", "")),
        "Drug Discovery Stage": detect_stage(p.get("abstract", "")),
        "Citations": str(p.get("citationCount", "")),
        "DOI": doi,
        "PubMed ID": pmid,
        "URL": url,
        "Open Access": "Yes" if p.get("isOpenAccess") else "No",
        "Publication Type": "; ".join(pub_types),
        "Abstract": p.get("abstract", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_papers: dict[str, dict] = {}  # s2 paper_id → paper dict

    def add(papers_list: list[dict], source: str):
        added = 0
        for p in papers_list:
            pid = p.get("paperId")
            if not pid or pid in all_papers:
                continue
            keep, reason = passes_filter(p)
            if keep:
                p["_source"] = source
                all_papers[pid] = p
                added += 1
        return added

    # 1. Look up known seed papers
    print("Step 1: Fetching known seed papers...")
    seed_ids = []
    for title in SEED_TITLES:
        p = s2_get_paper(title)
        if p:
            pid = p.get("paperId")
            if pid:
                seed_ids.append(pid)
                keep, reason = passes_filter(p)
                if keep:
                    p["_source"] = "seed"
                    all_papers[pid] = p
                    print(f"  ✓ {p.get('year')} — {p.get('title', '')[:70]}")
                else:
                    # Still keep seed papers even if they don't pass strict filter
                    p["_source"] = "seed"
                    all_papers[pid] = p
                    print(f"  ~ {p.get('year')} — {p.get('title', '')[:70]} [{reason}]")
        time.sleep(1)
    print(f"  Found {len(seed_ids)} seed papers")

    # 2. Semantic search queries
    print("\nStep 2: Semantic search...")
    for query in QUERIES:
        print(f"  Query: {query[:60]}...", end=" ", flush=True)
        results = s2_search(query, limit=100)
        n = add(results, f"search:{query[:40]}")
        print(f"→ {len(results)} results, {n} new passed filter")
        time.sleep(1)

    # 3. Recommendations from seed papers
    print(f"\nStep 3: Getting recommendations from {len(seed_ids)} seed papers...")
    recs = s2_recommendations(seed_ids)
    n = add(recs, "recommendations")
    print(f"  {len(recs)} recommendations → {n} new passed filter")
    time.sleep(1)

    # 4. Citations of seed papers
    print(f"\nStep 4: Getting citations of seed papers...")
    for pid in seed_ids[:10]:
        p = all_papers.get(pid, {})
        cits = s2_citations(pid, limit=100)
        n = add(cits, f"citations_of:{p.get('title','')[:30]}")
        if n:
            print(f"  +{n} from citations of: {p.get('title','')[:60]}")
        time.sleep(1)

    print(f"\nTotal verified papers: {len(all_papers)}")

    # Format and sort
    rows = [fmt_paper(p) for p in all_papers.values()]
    stage_order = {
        "Clinical / Approved": 0,
        "IND-enabling / Clinical Candidate": 1,
        "In Vivo Preclinical": 2,
        "Lead Optimisation": 3,
        "Hit Identification / In Vitro": 4,
        "Computational / Hit Discovery": 5,
        "Stage unclear": 6,
    }
    rows.sort(key=lambda x: (
        -int(x.get("Year") or 0),
        stage_order.get(x.get("Drug Discovery Stage", ""), 9)
    ))

    # Summary
    print("\nStage breakdown:")
    for stage, n in sorted(Counter(r["Drug Discovery Stage"] for r in rows).items(),
                            key=lambda x: stage_order.get(x[0], 9)):
        print(f"  {stage}: {n}")

    out = "/Users/joyal/lab-website-2/alphafold_semantic.csv"
    fields = [
        "Title", "Authors", "Journal", "Year",
        "Drug Discovery Stage", "Citations",
        "DOI", "PubMed ID", "URL", "Open Access",
        "Publication Type", "Abstract"
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
