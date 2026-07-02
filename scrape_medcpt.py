#!/usr/bin/env python3
"""
MedCPT-powered retrieval of AlphaFold small-molecule drug discovery papers.
Pipeline:
  1. Broad PubMed fetch (~3000 candidate abstracts)
  2. MedCPT-Article-Encoder  → article embeddings
     MedCPT-Query-Encoder    → query embedding
  3. Cosine similarity re-ranking
  4. Top papers re-scored with MedCPT-Cross-Encoder
  5. Filter: must have experimental results (IC50/Ki/in vivo/synthesis)
  6. Export CSV
"""

import csv, re, time, xml.etree.ElementTree as ET
from collections import Counter

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ── Queries ───────────────────────────────────────────────────────────────────

QUERY = (
    "(alphafold[tiab] OR alphafold2[tiab] OR alphafold3[tiab] OR "
    '"alpha fold"[tiab] OR AF2[tiab]) AND '
    '(inhibitor[tiab] OR "small molecule"[tiab] OR docking[tiab] OR '
    '"virtual screening"[tiab] OR "drug discovery"[tiab] OR '
    '"structure-based"[tiab] OR IC50[tiab] OR scaffold[tiab] OR '
    '"lead compound"[tiab] OR "hit compound"[tiab] OR '
    '"binding affinity"[tiab]) '
    'NOT Review[pt] NOT "Systematic Review"[pt]'
)

# Natural language query for MedCPT semantic matching
NL_QUERY = (
    "AlphaFold protein structure prediction used to discover "
    "novel small molecule inhibitors with experimental validation "
    "including synthesis and biological assays showing IC50 or binding affinity"
)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── PubMed helpers ─────────────────────────────────────────────────────────────

def pubmed_search(query, retmax=3000):
    r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params={
        "db": "pubmed", "term": query,
        "retmax": retmax, "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()["esearchresult"]
    return data.get("idlist", []), int(data.get("count", 0))

def pubmed_fetch(pmids, batch=200):
    papers = []
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i+batch]
        r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(chunk),
            "rettype": "xml", "retmode": "xml",
        }, timeout=60)
        r.raise_for_status()
        papers.extend(parse_xml(r.text))
        print(f"  Fetched {min(i+batch, len(pmids))}/{len(pmids)}", flush=True)
        time.sleep(0.35)
    return papers

def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        p = {}
        te = art.find(".//ArticleTitle")
        p["title"] = "".join(te.itertext()).strip() if te is not None else ""
        authors = [f"{a.findtext('LastName','')} {a.findtext('ForeName','')}".strip()
                   for a in art.findall(".//Author") if a.findtext("LastName")]
        p["authors"] = "; ".join(authors)
        je = art.find(".//Journal/Title")
        p["journal"] = je.text.strip() if je is not None else ""
        pd = art.find(".//PubDate")
        p["year"] = (pd.findtext("Year","") or pd.findtext("MedlineDate","")[:4]) if pd is not None else ""
        pmid_el = art.find(".//PMID")
        p["pmid"] = pmid_el.text.strip() if pmid_el is not None else ""
        doi = ""
        for eid in art.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text.strip(); break
        p["doi"] = doi
        p["url"] = (f"https://doi.org/{doi}" if doi else
                    f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/" if p["pmid"] else "")
        abs_parts = []
        for ab in art.findall(".//AbstractText"):
            lbl = ab.get("Label","")
            txt = "".join(ab.itertext()).strip()
            abs_parts.append(f"{lbl}: {txt}" if lbl else txt)
        p["abstract"] = " ".join(abs_parts)
        pt = [x.text for x in art.findall(".//PublicationType") if x.text]
        p["pub_types"] = "; ".join(pt)
        kws = [k.text.strip() for k in art.findall(".//Keyword") if k.text]
        mesh = [m.findtext("DescriptorName","").strip() for m in art.findall(".//MeshHeading")]
        p["keywords"] = "; ".join(kws + [m for m in mesh if m])
        out.append(p)
    return out

# ── Experimental evidence filter ──────────────────────────────────────────────

def has_experimental_evidence(abstract: str) -> bool:
    """Must have real wet-lab data, not just computational."""
    t = abstract.lower()
    return bool(re.search(
        r'(IC50|EC50|\bKi\b|\bKd\b|GI50|MIC\b|pIC50|inhibit.{0,20}\d+\s*[nμu]M|'
        r'bind.{0,20}\d+\s*[nμu]M|synthes|characteriz|biolog.{0,20}(assay|evaluat|test)|'
        r'in\s*vitro|enzymatic\s*assay|cell\s*(viability|proliferat)|'
        r'western\s*blot|flow\s*cytometr|selectivity\s*profil)', t))

def detect_stage(abstract: str) -> str:
    t = abstract.lower()
    if re.search(r'(phase\s*[123i]{1,3}|clinical\s*trial|first.in.human|'
                 r'fda.approved|ema.approved|entered\s*clinic)', t):
        return "Clinical / Approved"
    if re.search(r'(clinical\s*candidate|ind\s*application)', t):
        return "IND-enabling / Clinical Candidate"
    if re.search(r'(in\s*vivo|animal\s*model|mouse\s*model|rat\s*model|'
                 r'xenograft|pharmacokinetic|oral\s*bioavail|murine|'
                 r'tumor\s*(growth|regression|inhibit)|efficacy\s*in\s*(mice|rats))', t):
        return "In Vivo Preclinical"
    if re.search(r'(lead\s*optim|sar\s*(study|analys|explor)|structure.activity|'
                 r'second.round\s*design|analog\s*synthes|scaffold\s*hop)', t):
        return "Lead Optimisation"
    if re.search(r'(IC50|EC50|\bKi\b|\bKd\b|synthes|cell.{0,20}activ|'
                 r'cell.{0,20}assay|antiproliferat|enzymatic\s*assay)', t):
        return "Hit Identification / In Vitro"
    return "Computational"

def alphafold_used_as_tool(title: str, abstract: str) -> bool:
    text = (title + " " + abstract).lower()
    return bool(re.search(
        r'(alphafold.{0,60}(struct|model|predict|generat|database)|'
        r'(struct|model|predict).{0,60}alphafold|'
        r'(using|used|employ|utiliz|obtain).{0,30}alphafold|'
        r'alphafold.{0,30}(was used|were used|enabled)|'
        r'\baf2\b.{0,60}(struct|model|dock|screen)|'
        r'\baf3\b.{0,60}(struct|model|dock|screen))', text))

# ── MedCPT models ─────────────────────────────────────────────────────────────

def load_models():
    print("Loading MedCPT models...")
    q_tok  = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    q_mod  = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    a_tok  = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    a_mod  = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    cx_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    cx_mod = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    q_mod.eval(); a_mod.eval(); cx_mod.eval()
    print("Models loaded.")
    return q_tok, q_mod, a_tok, a_mod, cx_tok, cx_mod

def encode_query(query, q_tok, q_mod):
    with torch.no_grad():
        enc = q_tok([query], truncation=True, padding=True,
                    return_tensors="pt", max_length=64)
        emb = q_mod(**enc).last_hidden_state[:, 0, :]
    return F.normalize(emb, dim=-1)

def encode_articles(papers, a_tok, a_mod, batch=32):
    all_embs = []
    for i in range(0, len(papers), batch):
        chunk = papers[i:i+batch]
        pairs = [[p["title"], p["abstract"][:400]] for p in chunk]
        with torch.no_grad():
            enc = a_tok(pairs, truncation=True, padding=True,
                        return_tensors="pt", max_length=512)
            emb = a_mod(**enc).last_hidden_state[:, 0, :]
        all_embs.append(F.normalize(emb, dim=-1))
        if (i // batch) % 10 == 0:
            print(f"  Encoded {min(i+batch, len(papers))}/{len(papers)}", flush=True)
    return torch.cat(all_embs, dim=0)

def cross_encode(query, papers, cx_tok, cx_mod, batch=16):
    scores = []
    for i in range(0, len(papers), batch):
        chunk = papers[i:i+batch]
        pairs = [[query, f"{p['title']}. {p['abstract'][:300]}"] for p in chunk]
        with torch.no_grad():
            enc = cx_tok(pairs, truncation=True, padding=True,
                         return_tensors="pt", max_length=512)
            logits = cx_mod(**enc).logits.squeeze(dim=1)
        scores.extend(logits.tolist())
    return scores

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. PubMed broad fetch
    print("Step 1: Broad PubMed search...")
    pmids, count = pubmed_search(QUERY)
    print(f"  Found {count} papers, fetching {len(pmids)}")
    papers = pubmed_fetch(pmids)
    print(f"  Fetched {len(papers)} papers")

    # 2. Pre-filter: must have AlphaFold actually used + experimental evidence
    print("\nStep 2: Hard filtering (AlphaFold used as tool + experimental data)...")
    candidates = [p for p in papers
                  if alphafold_used_as_tool(p["title"], p["abstract"])
                  and has_experimental_evidence(p["abstract"])]
    print(f"  {len(candidates)} papers pass hard filter")

    if not candidates:
        print("No candidates found — check your PubMed query.")
        return

    # 3. Load MedCPT
    q_tok, q_mod, a_tok, a_mod, cx_tok, cx_mod = load_models()

    # 4. Bi-encoder ranking
    print("\nStep 3: MedCPT bi-encoder ranking...")
    q_emb = encode_query(NL_QUERY, q_tok, q_mod)
    a_embs = encode_articles(candidates, a_tok, a_mod)
    bi_scores = (a_embs @ q_emb.T).squeeze().tolist()
    if isinstance(bi_scores, float):
        bi_scores = [bi_scores]

    # Sort by bi-encoder score, take top 200 for cross-encoder
    sorted_idx = sorted(range(len(bi_scores)), key=lambda i: -bi_scores[i])
    ranked = [(bi_scores[i], candidates[i]) for i in sorted_idx]
    top_n = min(200, len(ranked))
    top_papers = [p for _, p in ranked[:top_n]]
    print(f"  Bi-encoder done, re-scoring top {top_n} with cross-encoder...")

    # 5. Cross-encoder re-ranking
    cx_scores = cross_encode(NL_QUERY, top_papers, cx_tok, cx_mod)
    cx_sorted_idx = sorted(range(len(cx_scores)), key=lambda i: -cx_scores[i])
    final_ranked = [(cx_scores[i], top_papers[i]) for i in cx_sorted_idx]

    # 6. Build output — keep all that passed hard filter, sorted by cross-encoder score
    rows = []
    for rank, (score, p) in enumerate(final_ranked, 1):
        rows.append({
            "Rank": rank,
            "MedCPT Score": f"{score:.3f}",
            "Title": p["title"],
            "Authors": p["authors"],
            "Journal": p["journal"],
            "Year": p["year"],
            "Drug Discovery Stage": detect_stage(p["abstract"]),
            "PMID": p["pmid"],
            "DOI": p["doi"],
            "URL": p["url"],
            "Publication Types": p["pub_types"],
            "Keywords": p["keywords"],
            "Abstract": p["abstract"],
        })

    # Also append papers that passed hard filter but weren't in top_n (bi-encoder only)
    top_pmids = {p["pmid"] for p in top_papers}
    for score, p in ranked[top_n:]:
        rows.append({
            "Rank": "",
            "MedCPT Score": f"{score:.3f}",
            "Title": p["title"],
            "Authors": p["authors"],
            "Journal": p["journal"],
            "Year": p["year"],
            "Drug Discovery Stage": detect_stage(p["abstract"]),
            "PMID": p["pmid"],
            "DOI": p["doi"],
            "URL": p["url"],
            "Publication Types": p["pub_types"],
            "Keywords": p["keywords"],
            "Abstract": p["abstract"],
        })

    # Stage summary
    print("\nStage breakdown:")
    order = ["Clinical / Approved","IND-enabling / Clinical Candidate",
             "In Vivo Preclinical","Lead Optimisation","Hit Identification / In Vitro","Computational"]
    for s in order:
        n = sum(1 for r in rows if r["Drug Discovery Stage"] == s)
        if n: print(f"  {s}: {n}")

    out = "/Users/joyal/lab-website-2/alphafold_medcpt.csv"
    fields = ["Rank","MedCPT Score","Title","Authors","Journal","Year",
              "Drug Discovery Stage","PMID","DOI","URL",
              "Publication Types","Keywords","Abstract"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone → {out}  ({len(rows)} papers, sorted by MedCPT relevance)")

if __name__ == "__main__":
    main()
