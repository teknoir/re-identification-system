# infer_embed.py
import argparse
from pathlib import Path
import numpy as np
import torch
from metric_model import EntryEncoder, CrossAttentionEntryEncoder
from data_loaders import EntryDataset

def load_model(ckpt_path: Path) -> EntryEncoder:
    ck = torch.load(ckpt_path, map_location="cpu")
    fusion_mode = ck.get("fusion_mode", "baseline")
    if fusion_mode == "xattn":
        m = CrossAttentionEntryEncoder(
            vis_dim=ck["vis_dim"],
            attr_dim=ck["attr_dim"],
            emb_dim=ck["emb_dim"],
            dropout=ck["dropout"],
        )
    else:
        m = EntryEncoder(
            ck["vis_dim"],
            ck["attr_dim"],
            ck["emb_dim"],
            ck.get("use_attention", False),  # Backwards compatibility
            ck["dropout"],
        )

    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--out-ids", type=Path, required=True)
    args = ap.parse_args()

    ds = EntryDataset(args.manifest, args.gt, args.schema)
    model = load_model(args.model)

    vecs, ids = [], []
    for i in range(len(ds)):
        item = ds[i]
        vis = item["vis"].unsqueeze(0).float()              # (1, N, Dv)
        mask = torch.ones((1, vis.shape[1]), dtype=torch.bool)
        attr = item["attr"].unsqueeze(0).float() if item["attr"].numel()>0 else None
        with torch.no_grad():
            z = model(vis, attr, mask=mask).squeeze(0).cpu().numpy()
        vecs.append(z); ids.append(item["entry_id"])

    V = np.stack(vecs, axis=0)
    np.save(args.out_npy, V)
    Path(args.out_ids).write_text("\n".join(ids))
    print(f"[done] wrote {args.out_npy} and {args.out_ids} ({V.shape[0]} entries, dim={V.shape[1]})")

if __name__ == "__main__":
    main()
