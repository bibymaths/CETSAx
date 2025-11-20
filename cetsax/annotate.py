#!/usr/bin/env python
"""
annotate.py
--------------------------------

Builds:
    1) protein_annotations.csv
    2) protein_sequences.fasta

from a list of protein IDs using mygene and UniProt REST, with parallelized
FASTA retrieval.

Usage example:

    python annotate.py \
        --fits-csv ec50_fits.csv \
        --id-col id \
        --species human \
        --out-annot protein_annotations.csv \
        --out-fasta protein_sequences.fasta \
        --max-workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import requests
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import mygene


# ---------------------------------------------------------------------
# 1. Get unique IDs from fits table
# ---------------------------------------------------------------------

def get_unique_ids(
        fits_csv: str | Path,
        id_col: str = "id",
) -> List[str]:
    df = pd.read_csv(fits_csv)
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in {fits_csv}")
    ids = df[id_col].dropna().astype(str).unique().tolist()
    return ids


# ---------------------------------------------------------------------
# Helper: strip isoform suffixes
# ---------------------------------------------------------------------
def strip_isoform_suffix(acc: str) -> str:
    """
    Remove UniProt isoform suffixes like '-1', '-2', '-10'.
    O00231-2 → O00231
    """
    if not isinstance(acc, str):
        return acc
    # Only strip final -number pattern
    if acc.count('-') == 1 and acc.split('-')[1].isdigit():
        return acc.split('-')[0]
    return acc


# ---------------------------------------------------------------------
# 2. Annotation via mygene
# ---------------------------------------------------------------------

def fetch_annotations_with_mygene(
        ids: List[str],
        species: str = "human",
        chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Use mygene to get basic annotations.

    We try multiple scopes: symbol, entrezgene, uniprot.
    Returned fields:
        symbol, name, entrezgene, uniprot, go.BP, pathway
    """
    mg = mygene.MyGeneInfo()

    all_records: List[Dict[str, Any]] = []

    # mygene is already "batch-parallelized" internally, so we just chunk
    for i in tqdm(range(0, len(ids), chunk_size), desc="mygene annotations"):
        batch = ids[i: i + chunk_size]

        res = mg.querymany(
            batch,
            scopes=["uniprot", "symbol", "entrezgene"],
            fields=[
                "symbol",
                "name",
                "entrezgene",
                "uniprot",
                "go.BP",
                "pathway",
            ],
            species=species,
            as_dataframe=False,
            returnall=False,
            verbose=False,
        )

        # res is a list of dicts
        for r in res:
            # 'query' is the original ID we asked for
            q = r.get("query")
            if q is None:
                continue

            symbol = r.get("symbol")
            name = r.get("name")
            entrez = r.get("entrezgene")
            # UniProt IDs may come as dict/list
            uni = r.get("uniprot")
            if isinstance(uni, dict):
                # e.g. {"Swiss-Prot": "P12345", "TrEMBL": [...]} or vice versa
                swiss = uni.get("Swiss-Prot")
                trembl = uni.get("TrEMBL")
                if isinstance(swiss, list):
                    swiss = swiss[0]
                if isinstance(trembl, list):
                    trembl = trembl[0]
                uniprot_acc = swiss or trembl
            elif isinstance(uni, list):
                uniprot_acc = uni[0]
            else:
                uniprot_acc = uni

            # GO BP terms
            go_bp = r.get("go", {}).get("BP", [])
            if isinstance(go_bp, dict):
                go_bp = [go_bp]
            bp_terms = []
            for bp in go_bp:
                term_id = bp.get("id")
                term_name = bp.get("term")
                if term_id and term_name:
                    bp_terms.append(f"{term_id}:{term_name}")
                elif term_name:
                    bp_terms.append(term_name)
            go_bp_str = ";".join(bp_terms) if bp_terms else ""

            # Pathway info (varies by source)
            pwy = r.get("pathway", {})
            p_terms: List[str] = []
            if isinstance(pwy, dict):
                # KEGG, Reactome, BioCarta etc.
                for src, val in pwy.items():
                    if isinstance(val, dict):
                        p_id = val.get("id")
                        p_name = val.get("name")
                        if p_id and p_name:
                            p_terms.append(f"{src}:{p_id}:{p_name}")
                        elif p_name:
                            p_terms.append(f"{src}:{p_name}")
                    elif isinstance(val, list):
                        for v in val:
                            p_id = v.get("id")
                            p_name = v.get("name")
                            if p_id and p_name:
                                p_terms.append(f"{src}:{p_id}:{p_name}")
                            elif p_name:
                                p_terms.append(f"{src}:{p_name}")
            pathway_str = ";".join(p_terms) if p_terms else ""

            all_records.append(
                {
                    "query_id": q,
                    "symbol": symbol,
                    "name": name,
                    "entrezgene": entrez,
                    "uniprot": uniprot_acc,
                    "go_bp": go_bp_str,
                    "pathway": pathway_str,
                }
            )

    annot_df = pd.DataFrame(all_records)

    # Deduplicate by query_id, keep the first best hit
    annot_df = (
        annot_df
        .drop_duplicates(subset=["query_id"])
        .rename(columns={"query_id": "id"})
    )

    return annot_df


# ---------------------------------------------------------------------
# 3. Parallel UniProt FASTA download
# ---------------------------------------------------------------------

def fetch_uniprot_fasta(acc: str, timeout: float = 10.0) -> Optional[str]:
    """
    Fetch a single UniProt FASTA entry by accession.

    Returns the FASTA string (with header + sequence) or None on failure.
    """
    if not isinstance(acc, str) or not acc:
        return None

    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text.startswith(">"):
            return r.text.strip()
        return None
    except Exception:
        return None


def fetch_fastas_parallel(
        accessions: List[str],
        max_workers: int = 8,
) -> Dict[str, str]:
    """
    Parallel UniProt FASTA retrieval for a list of accessions.

    Returns dict: {acc: fasta_text}.
    """
    accs = [a for a in sorted(set(accessions)) if isinstance(a, str) and a]

    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_uniprot_fasta, a): a for a in accs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="UniProt FASTA"):
            acc = futures[fut]
            fasta = fut.result()
            if fasta:
                results[acc] = fasta

    return results


def write_fastas(
        acc_to_fasta: Dict[str, str],
        out_fasta: str | Path,
) -> None:
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    with out_fasta.open("w") as fh:
        for acc, fasta in acc_to_fasta.items():
            # ensure header starts with accession
            # UniProt already does this, but we keep as-is.
            fh.write(fasta.strip() + "\n")

    print(f"Wrote {len(acc_to_fasta)} FASTA entries to {out_fasta}")


# ---------------------------------------------------------------------
# 4. Main: glue everything together
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build protein_annotations.csv and protein_sequences.fasta using mygene + UniProt (parallel)."
    )
    ap.add_argument(
        "--fits-csv",
        # required=True,
        default="../results/ec50_fits.csv",
        help="Input EC50 fits CSV with protein IDs (e.g. ec50_fits.csv).",
    )
    ap.add_argument(
        "--id-col",
        default="id",
        help="Column name for protein IDs in fits-csv.",
    )
    ap.add_argument(
        "--species",
        default="human",
        help="Species for mygene queries (e.g. 'human', 'mouse', 'rat').",
    )
    ap.add_argument(
        "--out-annot",
        default="protein_annotations.csv",
        help="Output CSV for protein annotations.",
    )
    ap.add_argument(
        "--out-fasta",
        default="protein_sequences.fasta",
        help="Output FASTA file for sequences.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of threads for parallel UniProt fetch.",
    )
    args = ap.parse_args()

    fits_path = Path(args.fits_csv)

    # 1) Get IDs
    ids = get_unique_ids(fits_path, id_col=args.id_col)
    ids = [strip_isoform_suffix(x) for x in ids]
    print(f"Found {len(ids)} unique IDs from {fits_path}")

    # 2) Annotations with mygene
    annot_df = fetch_annotations_with_mygene(
        ids,
        species=args.species,
    )

    # We want a protein-centric annotation table keyed by original id
    # If original id == uniprot accession, we can keep that as 'id'
    # Otherwise we still keep 'id' and add 'uniprot' column.
    out_annot_path = Path(args.out_annot)
    out_annot_path.parent.mkdir(parents=True, exist_ok=True)
    annot_df.to_csv(out_annot_path, index=False)
    print(f"Wrote annotations to {out_annot_path} (n={len(annot_df)})")

    # 3) Parallel FASTA download from UniProt
    # Priority: use uniprot column if present; otherwise original IDs.
    if "uniprot" in annot_df.columns:
        accs = annot_df["uniprot"].dropna().astype(str).tolist()
    else:
        accs = annot_df["id"].dropna().astype(str).tolist()

    acc_to_fasta = fetch_fastas_parallel(
        accessions=accs,
        max_workers=args.max_workers,
    )

    write_fastas(
        acc_to_fasta=acc_to_fasta,
        out_fasta=args.out_fasta,
    )

    print("Done.")


if __name__ == "__main__":
    main()
