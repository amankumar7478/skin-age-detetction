"""Write ids.txt (one FFHQ image number per line) from the 1,000 manual wrinkle masks.

    python scripts/03_make_id_list.py
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--masks", default="manual_wrinkle_masks")
ap.add_argument("--out", default="ids.txt")
a = ap.parse_args()

ids = sorted(int(p.stem) for p in Path(a.masks).glob("*.png"))
Path(a.out).write_text("\n".join(str(i) for i in ids) + "\n")
print(f"wrote {len(ids)} ids to {a.out} (first: {ids[:5]}, last: {ids[-3:]})")
