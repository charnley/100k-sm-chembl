import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
import sys
import io

DATA = Path("data")
MAIN = DATA / "DOWNLOAD-Z1ne6qrt4wu91Pko505qNL8HRHmP9w9SFGyvEarzIcM_eq_.csv"
PART2 = DATA / "DOWNLOAD-Z1ne6qrt4wu91Pko505qNL8HRHmP9w9SFGyvEarzIcM_eq__part2.csv"
OUT = Path("top100k_smallest.csv")

ID_COL = "Compound ChEMBL ID"
SMILES_COL = "Smiles"
HEAVY_COL = "Heavy Atoms"

COLS_MAIN = [ID_COL, SMILES_COL, HEAVY_COL]
TOP_K = 100_000
PRE_TAKE = 250_000
MIN_ATOMS = 2


def clean_salt(smiles: str) -> str:
    parts = smiles.split(".")
    if len(parts) == 1:
        return smiles
    return max(parts, key=len)


def validate_smiles(smi: str) -> tuple[bool, str, int]:
    # Capture RDKit warnings
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    RDLogger.logger().setLevel(RDLogger.WARNING)

    mol = Chem.MolFromSmiles(smi, sanitize=True)

    warnings = sys.stderr.getvalue()
    sys.stderr = old_stderr

    if mol is None:
        return False, "parse_fail", 0

    if warnings.strip():
        return False, "rdkit_warning", 0

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False, "sanitize_fail", 0

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return False, "radical", 0

    try:
        canon = Chem.MolToSmiles(mol, canonical=True)
        mol2 = Chem.MolFromSmiles(canon)
        if mol2 is None or mol2.GetNumAtoms() != mol.GetNumAtoms():
            return False, "roundtrip_fail", 0
    except Exception:
        return False, "roundtrip_fail", 0

    return True, "ok", mol.GetNumHeavyAtoms()


def main():
    print("Reading main CSV...")
    df_main = pd.read_csv(MAIN, sep=";", quotechar='"', usecols=COLS_MAIN)
    print(f"  main: {len(df_main):,} rows")

    print("Reading part2 CSV (no header)...")
    df_part2 = pd.read_csv(PART2, sep=";", quotechar='"', header=None, usecols=[0, 22, 19])
    df_part2.columns = COLS_MAIN
    print(f"  part2: {len(df_part2):,} rows")

    print("Merging...")
    df = pd.concat([df_main, df_part2], ignore_index=True)
    print(f"  total: {len(df):,} rows")

    print("Dropping rows with empty SMILES or Heavy Atoms...")
    before = len(df)
    df = df.dropna(subset=[SMILES_COL, HEAVY_COL])
    df[SMILES_COL] = df[SMILES_COL].astype(str)
    df[HEAVY_COL] = pd.to_numeric(df[HEAVY_COL], errors="coerce")
    df = df.dropna(subset=[HEAVY_COL])
    df[HEAVY_COL] = df[HEAVY_COL].astype(int)
    print(f"  dropped {before - len(df):,} rows")

    print(f"Pre-sorting by {HEAVY_COL} (fast), taking top {PRE_TAKE:,}...")
    df = df.sort_values(HEAVY_COL, ascending=True)
    df = df.head(PRE_TAKE).copy()
    print(f"  pre-sort range: {df[HEAVY_COL].min()} – {df[HEAVY_COL].max()}")

    print("Cleaning salts (on ~250K subset)...")
    df[SMILES_COL] = df[SMILES_COL].apply(clean_salt)

    print("Validating SMILES (no radicals, no RDKit warnings, round-trip)...")
    results = df[SMILES_COL].apply(validate_smiles)
    df["valid"] = results.apply(lambda x: x[0])
    df["fail_reason"] = results.apply(lambda x: x[1])
    df["num_atoms"] = results.apply(lambda x: x[2])

    fail_counts = df[~df["valid"]]["fail_reason"].value_counts()
    print("  Invalid breakdown:")
    for reason, count in fail_counts.items():
        print(f"    {reason}: {count:,}")

    valid_before = len(df)
    df = df[df["valid"]].copy()
    print(f"  dropped {valid_before - len(df):,} invalid, {len(df):,} valid")

    df["num_atoms"] = df["num_atoms"].astype(int)

    print("Dropping duplicate SMILES (keeping first)...")
    before = len(df)
    df = df.drop_duplicates(subset=[SMILES_COL], keep="first")
    print(f"  dropped {before - len(df):,} duplicates, {len(df):,} unique")

    print(f"Filtering: num_atoms >= {MIN_ATOMS}...")
    before = len(df)
    df = df[df["num_atoms"] >= MIN_ATOMS]
    print(f"  dropped {before - len(df):,} single-atom rows, {len(df):,} remain")

    print("Re-sorting by actual num_atoms...")
    df = df.sort_values("num_atoms", ascending=True)

    print(f"Taking top {TOP_K:,}...")
    df_top = df.head(TOP_K)
    assert ID_COL in df_top.columns and SMILES_COL in df_top.columns, f"Missing cols: {df_top.columns.tolist()}"

    print(f"Atom range: {df_top['num_atoms'].min()} – {df_top['num_atoms'].max()}")

    print(f"Exporting to {OUT}...")
    df_top[[ID_COL, SMILES_COL]].to_csv(OUT, index=False)

    verify = pd.read_csv(OUT)
    print(f"Output: {len(verify):,} rows, columns={verify.columns.tolist()}")
    print("Done.")


if __name__ == "__main__":
    main()
