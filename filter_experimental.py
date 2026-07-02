#!/usr/bin/env python3
"""
Post-filter alphafold_medcpt.csv to keep only papers with
genuine wet-lab experimental validation.
"""

import csv, re

def has_wetlab_evidence(abstract: str) -> tuple[bool, str]:
    t = abstract.lower()

    # ── Hard excludes: clearly computational-only language ─────────────────
    computational_only = [
        r'predict(ed)?\s+(IC50|binding|affinity|inhibit)',
        r'(MM[-\s]?GBSA|MM[-\s]?PBSA)',
        r'docking\s+score',
        r'free\s+energy\s+(calculat|predict)',
        r'only\s+(in\s+silico|computational)',
        r'no\s+(experimental|wet.?lab|in\s+vitro)',
    ]
    for pat in computational_only:
        if re.search(pat, t):
            return False, f"computational language: {pat}"

    # ── Must have at least ONE clear wet-lab signal ─────────────────────────

    # Measured activity values (not predicted)
    measured_activity = bool(re.search(
        r'(IC50|EC50|GI50|CC50|MIC|pIC50|pEC50|inhibitory\s+concentration)\s*'
        r'(of|=|value|was|were)?\s*[\d<>~≈]',
        abstract, re.IGNORECASE))

    # Binding constants from biophysical assays
    binding_assay = bool(re.search(
        r'(\bKi\b|\bKd\b|\bKD\b)\s*(of|=|value|was)?\s*[\d<>~]',
        abstract, re.IGNORECASE))

    # Explicit synthesis
    synthesis = bool(re.search(
        r'(was\s+synthes|were\s+synthes|synthes[ia]z|'
        r'compound(s)?\s+(were|was)\s+(prepared|made|obtained|synthesize)|'
        r'synthesis\s+of\s+compound|'
        r'total\s+synthesis|'
        r'we\s+(synthes|prepared|made))',
        t))

    # Cell-based assays
    cell_assay = bool(re.search(
        r'(cell\s+(viability|proliferation|cytotoxicity|growth\s+inhibit)|'
        r'antiproliferat|cytotoxic\s+activit|'
        r'MTT\s+assay|CCK.?8|MTS\s+assay|'
        r'western\s+blot|flow\s+cytometr|'
        r'apoptosis\s+(assay|induction|was\s+induced)|'
        r'colony\s+formation)',
        t))

    # Biophysical/structural validation
    biophysical = bool(re.search(
        r'(surface\s+plasmon\s+resonance|\bSPR\b|'
        r'isothermal\s+titration\s+calorimetr|\bITC\b|'
        r'thermal\s+shift\s+assay|\bDSF\b|'
        r'co.?crystal|crystal\s+structure\s+of|'
        r'cryo.?em\s+structure\s+of|'
        r'NMR\s+(confirm|validat|spectroscop))',
        t))

    # Enzymatic assays
    enzymatic = bool(re.search(
        r'(enzymatic\s+assay|enzyme\s+(inhibit|activit)|'
        r'kinase\s+assay|HDAC\s+assay|protease\s+assay|'
        r'inhibit(ed|ion|ory)\s+(activit|potenc)|'
        r'selectivity\s+(profil|against|over|toward))',
        t))

    # In vivo
    in_vivo = bool(re.search(
        r'(in\s+vivo|animal\s+model|mouse\s+model|rat\s+model|'
        r'xenograft|pharmacokinetic|oral\s+bioavailab|'
        r'tumor\s+(growth|regression|inhibit)|murine)',
        t))

    signals = {
        "measured activity": measured_activity,
        "binding assay": binding_assay,
        "synthesis": synthesis,
        "cell assay": cell_assay,
        "biophysical": biophysical,
        "enzymatic": enzymatic,
        "in vivo": in_vivo,
    }

    true_signals = [k for k, v in signals.items() if v]

    # Need synthesis OR biophysical/enzymatic assay + at least one activity signal
    has_activity = measured_activity or binding_assay or enzymatic
    has_lab = synthesis or cell_assay or biophysical or in_vivo or enzymatic

    if has_activity and has_lab:
        return True, ", ".join(true_signals)
    return False, f"only: {', '.join(true_signals) or 'none'}"


def detect_stage(abstract: str) -> str:
    t = abstract.lower()
    if re.search(r'(phase\s*[123i]{1,3}|clinical\s*trial|first.in.human|'
                 r'fda.approved|ema.approved|entered\s*clinic)', t):
        return "Clinical / Approved"
    if re.search(r'(clinical\s*candidate|ind\s*application)', t):
        return "IND-enabling / Clinical Candidate"
    if re.search(r'(in\s*vivo|animal\s*model|mouse\s*model|rat\s*model|'
                 r'xenograft|pharmacokinetic|oral\s*bioavail|murine|'
                 r'tumor\s*(growth|regression|inhibit)|'
                 r'efficacy\s*in\s*(mice|rats|vivo))', t):
        return "In Vivo Preclinical"
    if re.search(r'(lead\s*optim|sar\s*(study|analys|explor)|structure.activity|'
                 r'second.round|analog\s*synthes|scaffold\s*hop)', t):
        return "Lead Optimisation"
    if re.search(r'(IC50|EC50|\bKi\b|\bKd\b|cell\s*(viability|proliferat|assay)|'
                 r'antiproliferat|enzymatic\s*assay|western\s*blot|flow\s*cytometr)', t):
        return "Hit Identification / In Vitro"
    return "Computational"


def main():
    with open("/Users/joyal/lab-website-2/alphafold_medcpt.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    for r in rows:
        abstract = r.get("Abstract", "")
        ok, evidence = has_wetlab_evidence(abstract)
        if ok:
            r["Experimental Evidence"] = evidence
            r["Drug Discovery Stage"] = detect_stage(abstract)  # re-classify
            kept.append(r)

    # Remove computational-only that snuck through
    kept = [r for r in kept if r["Drug Discovery Stage"] != "Computational"]

    print(f"Kept {len(kept)} / {len(rows)} papers with genuine experimental validation\n")

    from collections import Counter
    order = ["Clinical / Approved", "IND-enabling / Clinical Candidate",
             "In Vivo Preclinical", "Lead Optimisation", "Hit Identification / In Vitro"]
    for stage in order:
        n = sum(1 for r in kept if r["Drug Discovery Stage"] == stage)
        if n:
            print(f"  {stage}: {n}")

    print("\nTop 20 by MedCPT score:")
    for r in kept[:20]:
        print(f"  [{r['Rank']:>3}] {r['MedCPT Score']}  [{r['Year']}]  {r['Drug Discovery Stage']}")
        print(f"       {r['Title'][:85]}")
        print(f"       Evidence: {r['Experimental Evidence']}")
        print()

    out = "/Users/joyal/lab-website-2/alphafold_experimental.csv"
    fields = ["Rank", "MedCPT Score", "Title", "Authors", "Journal", "Year",
              "Drug Discovery Stage", "Experimental Evidence",
              "PMID", "DOI", "URL", "Publication Types", "Keywords", "Abstract"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
