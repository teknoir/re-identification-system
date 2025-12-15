The pairs viewer is a system that helps clean the ground truth/training data for the entryencoder model by scanning for similarities to surface likely mistakes (high-confidence mismatches/positives) for manual review. It presents "high risk" pairs that we can use to highlight potential errors.

It looks for:
* hard positives: same pid but low similarity (suspicious merges)
* hard negatives: different pid but high similarity (suspicious splits / aliases)
* alias candidates: same store_id, different day_id, very high similarity (e.g. ≥ 0.93)

Today, we train the a metric encoder then use "infer_embed" to turn the trained checkpoint (model.pt) into reusable embeddings for every entry in the manifest. The training process has a few steps:
* train_metric2.py ...: Trains the metric encoder using PK sampling on the provided manifest/GT/schema
* infer_embed.py ...: Loads the trained checkpoint plus manifest/GT/schema to build an EntryDataset. Encodes every entry to a normalized
    embedding. Saves vectors to a NumPy file and entry IDs to a text file - entry_vectors.npy: fused entry embeddings, entry_ids.txt 
* export_strict_encoder_dataset.py ...: exports manifest.json: full line-crossing data, multi_gt.json: ground truth mapping of person entities to entries. These are pulled from mongo and are curated from production data using the manifest-editor 
* find_high_risk_pairs.py ...: Computes similarity pairs using embeddings/IDs with the manifest/GT. Flags likely mistakes by listing very-similar different-
    IDs and low-similar same-IDs based on provided thresholds/top-k. Writes candidate pairs to gt for manual review.

We still need to export ground truth for training purposes but I want to move the high risk pair editing to use mongo. So:
* We curate and cluster entries using the manifest editor
* Instead of exporting ground truth then running the find_high_risk_pairs.py script, I want the pairs viewer to modify the gt_tools.entries collection directly.

Look at the export_strict_encoder_dataset.py to see how we select ground truth. Instead of using the pairs_hard.jsonl, we should use mongo. We can update the gt_tools collections directly (toggling the include bool) but we also need to track which pairs we've already reviewed. So for each pair, we want an exclude button on both entry a and entry b. If we choose to exclude, say entry b, we'd modify the gt_tools collection accordingly. We'd also want to store that we've adjudicated that pair.


```bash
python manifest-editor/pairs_viewer/infer_embed.py \
    --manifest manifest-editor/pairs_viewer/entry.json \
    --gt manifest-editor/pairs_viewer/multi_gt.json \
    --schema matching-service/attr_schema.json \
    --model matching-service/models/encoder/model.pt \
    --out-npy matching-service/models/encoder/entry_vectors.npy \
    --out-ids matching-service/models/encoder/entry_ids.txt 
```


This is run against the ground truth data (multi_gt.json and entry.json)


```bash

export MODEL_RUN=xattn_pk_64_2_v1

python manifest-editor/pairs_viewer/find_high_risk_pairs.py \
  --vecs matching-service/models/encoder/entry_vectors.npy \
  --ids matching-service/models/encoder/entry_ids.txt \
  --manifest manifest-editor/pairs_viewer/entry.json \
  --gt manifest-editor/pairs_viewer/multi_gt.json \
  --neg-threshold 0.86 \
  --pos-threshold 0.70 \
  --topk-neg 20 \
  --output-dir manifest-editor/pairs_viewer 
```


  

# High-Risk Pair Viewer

This is a simple web-based tool to view high-risk pairs of images for ground truth verification.

## Installation

Install the required Python packages from the project's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Server

To start the web server, you need to set the following environment variables:

- `PAIRS_JSONL`: Path to the JSONL file containing the high-risk pairs.
- `MANIFEST_JSON`: Path to the JSON file containing the manifest of all entries.

Once the environment variables are set, you can start the server using `uvicorn`:

```bash
export PAIRS_JSONL=gt/pairs_hard.jsonl
export MANIFEST_JSON=model/encoder/runs/entry.json
uvicorn gt.pair_viewer.server:app --host 0.0.0.0 --port 9001 --reload
```

## Using the Viewer

After starting the server, open your web browser and navigate to:

[http://localhost:9001/static/index.html](http://localhost:9001/static/index.html)

You can then filter the pairs by risk type and sort them by similarity.

## Running inside the manifest editor server

The manifest editor FastAPI service now mounts this viewer under `/pairs-viewer` (UI at `/pairs-viewer/static/index.html`) so you can run a single server and browse pairs without starting a separate uvicorn instance. Make sure the viewer env vars (`PAIRS_JSONL`, `MANIFEST_JSON`, `MANIFEST_EDITOR_MONGO`, `MANIFEST_EDITOR_DB`) are set before launching the manifest editor service so the pairs viewer loads data from the expected files and Mongo database.

## Mongo-backed pairs (no jsonl)

`find_high_risk_pairs.py` can now write directly to Mongo and skip files:

```bash
python manifest-editor/pairs_viewer/find_high_risk_pairs.py \
  --vecs model/encoder/runs/<MODEL_RUN>/entry_vectors.npy \
  --ids model/encoder/runs/<MODEL_RUN>/entry_ids.txt \
  --manifest model/encoder/runs/entry.json \
  --gt model/encoder/runs/multi_gt.json \
  --neg-threshold 0.88 \
  --pos-threshold 0.75 \
  --topk-neg 20 \
  --mongo-uri "$MANIFEST_EDITOR_MONGO" \
  --mongo-db gt_tools \
  --mongo-coll high_risk_pairs \
  --mongo-drop \
  --run-id <MODEL_RUN> \
  --no-files
```

The pairs viewer will preferentially load from the `gt_tools.high_risk_pairs` collection (set `PAIRS_COLLECTION` to override) and fall back to `PAIRS_JSONL` if Mongo is empty/unavailable.
