#!/usr/bin/env python3
"""Download the ClothTransformer dataset from Hugging Face.

    python scripts/download_data.py                         # full dataset (~33 GB)
    python scripts/download_data.py --split test            # test split only (~3.3 GB)
    python scripts/download_data.py --file-list examples/showcase.txt

Only the ``sim_XXXXX.npz`` files are downloaded; the optional rest meshes with
UVs (``*_uv.obj`` / ``*_uv.bgeo.sc``) are skipped unless ``--with-uv-meshes``.
"""
import argparse
import json
import os

from huggingface_hub import snapshot_download

REPO_ID = "YuCrazing/ClothTransformer-dataset"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "ClothTransformer-dataset"),
                        help="target directory (default: data/ClothTransformer-dataset)")
    parser.add_argument("--split", choices=["train", "val", "test"], nargs="+",
                        help="only download these splits of splits/ct_split.json")
    parser.add_argument("--category", nargs="+",
                        help="only download these scenarios, e.g. sims_garment_anim_all")
    parser.add_argument("--file-list", help="text file of <category>/<sim_id> lines to download")
    parser.add_argument("--with-uv-meshes", action="store_true", help="also download the rest meshes with UVs")
    args = parser.parse_args()

    if args.file_list:
        with open(args.file_list) as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        patterns = [f"{i}.npz" for i in ids]
        if args.with_uv_meshes:
            patterns += [f"{i}_processed_cloth_*" for i in ids]
    elif args.split:
        with open(os.path.join(REPO_ROOT, "splits", "ct_split.json")) as f:
            split = json.load(f)
        ids = [i for s in args.split for i in split[s]]
        if args.category:
            ids = [i for i in ids if i.split("/")[0] in args.category]
        patterns = [f"{i}.npz" for i in ids]
        if args.with_uv_meshes:
            patterns += [f"{i}_processed_cloth_*" for i in ids]
    else:
        cats = args.category or ["*"]
        patterns = [f"{c}/sim_*.npz" for c in cats]
        if args.with_uv_meshes:
            patterns += [f"{c}/sim_*_processed_cloth_*" for c in cats]

    patterns += ["README.md"]
    print(f"Downloading {REPO_ID} -> {args.out}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=args.out, allow_patterns=patterns)
    print("Done.")


if __name__ == "__main__":
    main()
