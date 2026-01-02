# Re-entry Matching Service (FAISS + Metric Encoder)


The re-entry service attempts to cluster together entries (really any line-crossings in either direction) from the same store and the same day. 
  - Scope: /match and /manifest operate within a single (store_id, day_id) bucket. The FAISS index and clustering are per store-day; nothing crosses days or stores.
  - Inputs: It will index whatever entries you POST (typically line-crossing events) regardless of direction (entry/exit), as long as the payload has the matching store_id and
    day_id.
  - Clustering: The “clustering” is implicit—matching within that bucket and grouping entries into people for /manifest; it doesn’t attempt cross-day or cross-store clustering.

The primary algorithm is an entry encoder that fuses together visual embedding and attribute embedding similarities. If the fused score is above the threshold/margin, these entry events are clustered again signifying the same person.
  - The entry encoder produces a fused embedding from visual + attribute inputs; matching uses cosine similarity of those fused vectors.
  - The runtime rule is threshold + margin on the top-2 neighbors: match if nn1 >= THRESHOLD and (nn1 - nn2) >= MARGIN. 
  - The “clustering” is just grouping matched entries within the same store/day; there’s no separate clustering pass—each /match result is appended to the in-memory FAISS index
    for that bucket and /manifest reads those groupings.


## Production pipeline
Our edge devices run a Deepstream pipeline which supports configuration of virtual line-crossings. The general flow is:
- The object detector detects a new `person` class instance and assigns a object id (by the object tracker)
- When the person crosses the line-crossing, we generate ~10 "cutouts" of that person. Runtime sampling attempts to provide a good distrubition, optimizing for image quality and diversity in the images. 
- This event is then sent to the observatory processing pipline (OBS)
- The OBS pipeline sends these ~10 images are sent to an image embedding service ([SWIN-based transformer](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/re_identification_transformer/re_identification_transformer.html)). The SWIN-based transformer has been finetuned on 100K+ images from the customer environment (sampled across all cameras and stores). We generate separate embeddings for each image. 
- The embeddings are filtered by the OBS pipeline (and during training), see README_frame_filtering.md for extensive detail on this process
- The same ~10 images are also sent to a finetuned gemini endpoint to create attribute embeddings. The OBS pipeline sends images to our `llm_service` which sends them to our finetuned gemini endpoint. This endpoint acts as an intelligent "describer" that takes one or more images of a person and outputs a strictly defined JSON object containing clothing attributes. It uses the instructor library to force the llm to adhere to a specific Pydantic schema (Attrs), ensuring reliable, structured data instead of free-form text.

```
 "attrs": {
      "outerwear": "none",
      "outerwear_color": "unknown",
      "outerwear_pattern": "unknown",
      ...
      "one_piece_confidence": 0.1,
      "outerwear_confidence": 0.1,
      "top_wear_confidence": 1,
      "bottom_wear_confidence": 1,
      "footwear_confidence": 1,
      "footwear_color_confidence": 1,
```

- To support different use cases, we use facerec only on employees, only in the back rooms (there's a virtual line crossing at each back room entrance). If a line-crossing comes from one of these camers, we send the cutout stream (of ~10 images) to the facerec endpoint and store the matching employee id
- The line-crossing event is sent to the `matching-service` `/match` endpoint with results stored in `reid_service.observations`. The match data along with all event data is persisted to mongo in the `historian.line-crossings collection`


## Manifests
A “manifest” is the clustered view of all entries for a given (store_id, day_id): it groups related entry events (line crossings) into people, with their metadata (timestamp, camera, direction, images, attrs, etc.) and the match status derived from the matcher.
  - `/manifest` (served by the `matching-service`): Live view from the matcher. It queries the in-memory FAISS bucket for that store/day, applies the match rule, and returns the clustered people. It can filter by entry_id/camera and includes embeddings/images unless stripped.
  - `/manifest-proxy` (served by the `manifest-editor`): A thin wrapper that fetches the manifest, strips heavy fields (embeddings/images) for editor use, supports additional filters (entry_lookup, emp_id), and can resolve day/store from entry_lookup via gt_tools.clusters. It’s optimized for the manifest editor UI (lighter payload, optional employee expansion) rather than raw matcher output. *This endpoing is used to serve out cleansed/corrected manifests for the timecard fraud use case. It allows us to fix per-employee manifests by manually fixing clustering issues. Think of /manifest-proxy as viewing manifests through corrective lenses

## Manifest Editor
The manifest editor is used for two primary purposes:
- Mining line-crossing events for inclusion in the ground truth dataset
- Fixing "manifests"

It supports:
- Reassigning line-crossing events ("entries") between clusters as well as splitting out individual clusters
- Aliases to aid in labeling (though not used in training)
- Setting an employee id

In the case of ground truth mining, we only ever want to have a single cluster for a given person. If we have the same person multiple times in the training data, we're telling the algo during training that they are not the same person which causes issues. We "exclude" clusters:
- When the same person is present in multiple stores or on multiple days
- Employees, delivery people, cash carriers are all frequent occupants. We work hard to ensure only one instance of each person is present in the dataset

## Hard pairs viewer

The pairs viewer is a system that helps clean the ground truth/training data for the entryencoder model by scanning for similarities to surface likely mistakes (high-confidence mismatches/positives) for manual review. It presents "high risk" pairs that we can use to highlight potential errors. The "hard pairs" are automatically updated during the entry encoder training process.

It looks for:
* hard positives: same pid but low similarity (suspicious merges)
* hard negatives: different pid but high similarity (suspicious splits / aliases)
* alias candidates: same store_id, different day_id, very high similarity (e.g. ≥ 0.93)

- Data source: gt_tools.high_risk_pair. Images resolved via manifest-editor image proxy; artifacts vis_sim/entry_sims are rendered.
- UI location: /manifest-editor/pairs-viewer/static/index.html (same FastAPI app; port 8883 in local dev via run_local.sh).
- Controls: risk filter (default alias candidates), store/day filters, sort, page size, update button reloads pairs from Mongo.
- Visualization: two entries + heatmap of artifacts.vis_sim (row/col labels) and sim summary (vis_pooled, attr), badges for artifact presence.
- Actions: exclude person (sets persons.{pid}.include=false in gt_tools.clusters), dismiss/mark reviewed (reviewed_pairs in Mongo).

Features:
- Displays each `(qid, aid)` pair side-by-side with lazy-loaded thumbnails fetched via `/api/image` (backed by `visual_embeddings.gcs.download_blob_bytes` for `gs://` URIs).
- Keyboard-driven workflow: `j`/`k` move to the next/previous card, `space` toggles keep/remove, `u` marks a pair back to “unreviewed”.
- Decisions persist in `PAIRS_STATE_PATH`, so you can resume later or work in batches; use the filter dropdown to focus on unreviewed or previously removed pairs.
- Cards flagged for removal are highlighted, giving you a last visual check before dropping them from the ground-truth set. Make sure `PAIRS_IMAGE_BASE` points to the folder or bucket that contains the `person_*` image directories (and set `GOOGLE_APPLICATION_CREDENTIALS` if you point to GCS) so the thumbnails can be fetched.


# EntryEncoder

Siamese/Triplet encoder `f(entry)` that fuses multi-image visual embeddings and schema-vectorized attributes.

## Architecture Overview

The EntryEncoder is a deep learning model designed for person re-identification (ReID) across multiple camera views. It combines visual features from multiple images of the same person with attribute information to create a unified embedding suitable for similarity matching.

### Core Components

#### 1. **Visual Feature Processing**
- **Input**: Multiple pre-computed visual embeddings per entry (from burst sequences)
  - Each entry contains ~10 image embeddings from a line-crossing event
  - Visual embeddings are L2-normalized vectors (1024 dims from a SWIN-based reID model)

- **Aggregation Strategy**: Two modes available:
  - **Attention Pooling** (recommended): Uses `AttentiveSetPool` to intelligently combine multiple frames
  - **Mean Pooling**: Simple averaging with masking support for variable-length sequences

#### 2. **Attribute Feature Processing**
- **Input**: Structured attribute vector derived from VLM-extracted clothing attributes (SFT gemini endpoint trained on attribute ground truth dataset)
- **Schema Encoding**: Converts human-readable attributes to numerical vectors using:
  - **Enum fields**: One-hot encoding (e.g., `top_wear: t-shirt` → `[0,1,0,0,...]`)
  - **Multilabel fields**: Multi-hot encoding for multiple values
  - **Probability distributions**: Normalized confidence scores
  - **Float/bool fields**: Scalar values normalized to [0,1]

#### 3. **Feature Fusion**
- Visual and attribute features are projected to the same embedding dimension
- Concatenated and passed through a fusion network
- Final output is L2-normalized to unit sphere

### Detailed Component Breakdown

#### AttentiveSetPool Module
```
Input: (B, N, D) tensor of N frame embeddings
1. Query generation: q = Linear(first_frame) → (B, 1, H)
2. Key generation: k = Linear(all_frames) → (B, N, H)
3. Value generation: v = Linear(all_frames) → (B, N, D)
4. Attention scores: scores = q @ k^T * scale
5. Apply mask for variable-length sequences
6. Weighted combination: output = softmax(scores) @ v
```

This allows the model to focus on the most informative frames (e.g., frontal views) while handling occlusions and motion blur.

#### EntryEncoder Architecture
```
Visual Path:
  Multi-frame embeddings (N × Dv)
    ↓
  AttentiveSetPool/MeanPool
    ↓
  Linear(Dv → emb_dim)

Attribute Path:
  Attribute vector (Da)
    ↓
  Linear(Da → emb_dim)

Fusion:
  [Visual_emb, Attr_emb]
    ↓
  Linear(2×emb_dim → emb_dim) + ReLU + Dropout
    ↓
  Linear(emb_dim → emb_dim)
    ↓
  L2 Normalization
```

### Training Strategy

#### PK Sampling
The model uses a sophisticated batch construction strategy:
- **P identities × K samples** per batch (e.g., P=32, K=4 → batch_size=128)
- Ensures every person in the batch has multiple examples for comparison
- Enables hard negative mining within each batch

#### Batch Hard Triplet Loss
```python
For each anchor in the batch:
1. Find hardest positive: Same person, most distant embedding
2. Find hardest negative: Different person, closest embedding
3. Loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
```

This loss function focuses training on the most challenging cases, improving discrimination at decision boundaries.

### Key Design Decisions

1. **Multi-frame fusion**: Critical for handling partial occlusions and varying viewpoints
2. **Attention mechanism**: Learns to weight frames by quality/informativeness
3. **Attribute integration**: Provides semantic regularization and handles visual ambiguity
4. **L2 normalization**: Enables cosine similarity for fast retrieval
5. **Hard triplet mining**: Focuses learning on difficult cases near the decision boundary

### Inference Pipeline

1. **Feature Extraction**:
   - Load burst sequence images
   - Extract visual embeddings using pre-trained model
   - Extract attributes using VLM service

2. **Encoding**:
   - Pass visual embeddings + attributes through EntryEncoder
   - Output: 128-dimensional L2-normalized vector

3. **Matching**:
   - Query FAISS index (one per day) with K=20 nearest neighbors
   - Apply runtime rule on the top-2 neighbors (NN1, NN2):
   - Match if (NN1 >= 0.88) and ((NN1 - NN2) >= 0.02)
   - Otherwise: New person

### Performance Considerations

- **Embedding dimension**: 128 provides good balance of accuracy and speed
- **Attention pooling**: ~5% accuracy improvement over mean pooling
- **Attribute features**: ~3-7% improvement, especially for similar-looking people
- **Hard triplet loss**: Faster convergence and better boundary discrimination than contrastive loss



# Re-entry Matching Service Detail

This FastAPI service exposes a **query-then-add** endpoint that:
1) encodes a new entry (multi-image ReID vectors + attrs) into a fused vector,
2) queries today's FAISS index (top-K cosine),
3) applies the runtime rule:
   - `match if (NN1 >= THRESHOLD) and ((NN1 - NN2) >= MARGIN)`,
4) adds the entry to the index,
5) (optionally) persists raw inputs to Mongo.

## Requirements
- A trained encoder checkpoint (`model.pt`) saved with:
  - `vis_dim`, `attr_dim`, `emb_dim`, `use_attention`, `dropout`, `state_dict`
- Attribute schema JSON (`attr_schema.json`) matching your normalized attrs.
- A matching `frame_filtery.py` configuration (matched to training)

## Configure (env vars)
- `MODEL_CKPT` (default: `runs/metric_pk_v1/model.pt`)
- `ATTR_SCHEMA` (default: `attr_schema.json`)
- `MONGO_URI` (optional; empty disables Mongo persistence)
- `MONGO_DB` (default: `retail_reid`)
- `MONGO_EVENTS_COLLECTION` (default: `line-crossings`, used by `/manifest` to fetch detection metadata)
- `THRESHOLD` (default: `0.88`) <- this value is chosen from the model training logs
- `MARGIN` (default: `0.02`)
- `TOPK` (default: `20`)

*NOTE* The configuration of `frame_filter.py` should match the configuration of `frame_filter.py` used in training


## Running the API

Install dependencies (inside a virtualenv):
```bash
python -m pip install -r requirements.txt
```

Ensure your model checkpoint and schema paths are set (or use the defaults under `runs/` and `attr_schema.json`). If you want Mongo persistence, export `MONGO_URI` (and optionally `MONGO_DB`).

Start everthing:
```bash
./run_local.sh
```

## Training the Metric Encoder

Train the metric model (entry encoder)

```bash
model/encoder/train_encoder.sh
```

## Endpoints

### `GET /health`
Returns threshold/margin/topk + OK.

### `POST /match`
Query-then-add for a single entry. Matching happens only within the same `(day_id, store_id)` bucket. Embeddings are required; images/attrs are optional.

- Runtime rule: `status=match` only if `nn1 >= THRESHOLD` and `(nn1 - nn2) >= MARGIN` (top-2 neighbors from FAISS on the store/day bucket).
- Response includes `employee_id` when a face match is found (`historian.faces`) or when the matched observation already has one; otherwise null.

**Body (fields of interest)**
```json
{
  "day_id": "2025-11-06",
  "store_id": "nc0211",
  "entry_id": "nc0211-front-door-1-abc123",
  "alert_id": "64f8d1f42b7ed12f8a6d1234",
  "timestamp": "2025-11-06T15:02:11.200Z",
  "direction": "entry",
  "camera": "nc0211-front-door-1",
  "images": [
    "gs://victra-poc.teknoir.cloud/media/lc-person-cutouts/2025-11-06/வுகளில்",
    "gs://victra-poc.teknoir.cloud/media/lc-person-cutouts/2025-11-06/வுகளில்"
  ],
  "embeddings": [[...1024 floats...], [...], ...],
  "attrs": { "outerwear": "jacket", "outerwear_color": "red", "...": "..." },
  "topk": 20,
  "persist": true
}
```

**Response**
```json
{
  "ok": true,
  "timestamp": "2025-11-06T15:02:11.200Z",
  "direction": "entry",
  "camera": "nc0211-front-door-1",
  "images": [
    "gs://victra-poc.teknoir.cloud/media/lc-person-cutouts/2025-11-06/வுகளில்",
    "gs://victra-poc.teknoir.cloud/media/lc-person-cutouts/2025-11-06/வுகளில்"
  ],
  "embeddings": [[...],[...]],
  "status": "match",
  "match_id": "nc0211-front-door-1-xyz456",
  "score": 0.9143,
  "score2": 0.8921
}
```

### `POST /rebuild`
Body: `{ "day_id": "2025-11-06" }`

Clears the FAISS bucket for the specified day and repopulates it from MongoDB (`entries` collection) by re-encoding every persisted entry. Use this after restarting the service (or whenever you want to discard in-memory state) to reload the relevant day(s) before serving `/match`. Returns `{ "ok": true, "count": N }` on success.

### `GET /manifest`
Query params:
- `day_id` (required)
- `store_id` (required)
- `entry_id` (optional; limiting response to the person containing that entry)
- `camera` (optional, repeatable) filters events to specific camera names

Returns the clustered manifest for the given day/store, using the same FAISS similarity rule as `/match`. Each person entry includes `person_id`, `first_seen`, an ordered list of events (entry metadata + embeddings/images/scores), and can be filtered down to a single entry via `entry_id`.

**Example**
```bash
curl -G "http://0.0.0.0:8080/manifest" \
  --data-urlencode "day_id=2025-11-06" \
  --data-urlencode "store_id=nc0009" \
  --data-urlencode "camera=nc0009-front-door" \
  --data-urlencode "camera=nc0009-back-door" \
  --data-urlencode "entry_id=" > manifest.json
```

```
GET /manifest?day_id=2025-11-06&store_id=nc0009
```

```json
{
  "ok": true,
  "day_id": "2025-11-06",
  "store_id": "nc0009",
  "person_count": 2,
  "event_count": 5,
  "people": [
    {
      "person_id": "nc0009-0001",
      "first_seen": "2025-11-06T15:02:11.200Z",
      "events": [
        {
          "entry_id": "nc0009-front-door-123",
          "timestamp": "2025-11-06T15:02:11.200Z",
          "direction": "entry",
          "camera": "nc0009-front-door",
          "alert_id": "652f08a16f0ffe0001c9abcd",
          "images": ["gs://.../entry-0.jpg"],
          "score": null,
          "score2": null,
          "attrs": { "outerwear": "jacket", "skin_tone": "medium" },
          "embeddings": [[...]]
        },
        {
          "entry_id": "nc0009-front-door-456",
          "timestamp": "2025-11-06T19:42:03.018Z",
          "direction": "exit",
          "camera": "nc0009-front-door",
          "alert_id": "652f08a16f0ffe0001c9abce",
          "images": ["gs://.../exit-0.jpg"],
          "score": 0.913,
          "score2": 0.874,
          "attrs": { "outerwear": "jacket", "skin_tone": "medium" },
          "embeddings": [[...]]
        }
      ]
    },
    {
      "person_id": "nc0009-0002",
      "first_seen": "2025-11-06T17:10:55.801Z",
      "events": [
        {
          "entry_id": "nc0009-backroom-safe-789",
          "timestamp": "2025-11-06T17:10:55.801Z",
          "direction": "entry",
          "camera": "nc0009-backroom-safe-180",
          "alert_id": "652f08a16f0ffe0001c9abcf",
          "images": ["gs://.../safe-0.jpg"],
          "score": null,
          "score2": null,
          "attrs": { "outerwear": "hoodie", "skin_tone": "dark" },
          "embeddings": [[...]]
        }
      ]
    }
  ]
}
```

## Tooling

### Combined entry pipeline
Generates embeddings + VLM attrs and emits a ready-to-post `/match` payload (only required for local dev, the prod pipeline handles all of this automagically):

```bash
python entry_pipeline/process_entry.py \
  --mongo-uri "mongodb://teknoir:change-me@localhost:27017/historian?authSource=admin" \
  --mongo-collection line-crossings \
  --alerts-collection alerts \
  --entries-collection entries \
  --bucket gs://victra-poc.teknoir.cloud \
  --gcs-creds visual_embeddings/teknoir-c5ec5ebce12d.json \
  --reid-endpoint http://localhost:8081 \
  --vlm-model "projects/815276040543/locations/us-central1/endpoints/3211426074617446400" \
  --output 2025-11-06/nc0211/nc0211-front-door-2/entry/nc0211-front-door-2-8c511b69-12794.json
```

Batch version (process all cameras matching the prefix, in this case all cams for `nc0009`):
```bash
python entry_pipeline/batch_process_entries.py \
  --prefix nc0009 \
  --mongo-uri "mongodb://teknoir:change-me@localhost:27017/historian?authSource=admin" \
  --mongo-collection line-crossings \
  --alerts-collection alerts \
  --entries-collection entries \
  --bucket gs://victra-poc.teknoir.cloud \
  --gcs-creds visual_embeddings/teknoir-c5ec5ebce12d.json \
  --reid-endpoint http://localhost:8081 \
  --vlm-model "projects/815276040543/locations/us-central1/endpoints/3211426074617446400" \
  --day 2025-11-06 \
  --output-root payloads \
  --skip-existing
```

### Posting to `/match`
Use CURL for ad-hoc testing:
```bash
curl -X POST http://0.0.0.0:8080/match \
  -H "Content-Type: application/json" \
  --data-binary @payloads/2025-11-06/nc0211/nc0211-front-door-2/entry/nc0211-front-door-2-8c511b69-12794.json
```
For batches of payloads, use the helper script:
```bash
python entry_pipeline/post_payloads.py payloads/2025-11-06 \
  --match-url http://0.0.0.0:8080/match \
  --timeout 15
# add --dry-run to preview without posting
```



### Local test MongoDB
For local/development runs of `entry_pipeline/post_payloads.py` you can spin up an isolated Mongo instance:
```bash
docker compose -f docker-compose.mongo-test.yml up -d
# connection string (matches app.py defaults):
export MONGO_URI='mongodb://tester:testerpass@localhost:27018/retail_reid?authSource=admin'
```

To restart cleanly (data wiped):
```bash
docker compose -f docker-compose.mongo-test.yml down -v; docker compose -f docker-compose.mongo-test.yml up -d
```


## End-to-end Flow

1. **Data collection.**
   - Cameras publish `DetectionEvent` documents into Mongo (`line-crossings` collection). Each document contains `metadata.id`, `metadata.timestamp`, `data.peripheral.name` (camera), `data.burst` image keys, and `metadata.annotations["teknoir.org/linedir"]`.

2. **Payload generation.**
   - Run `entry_pipeline/process_entry.py` for a single entry or `entry_pipeline/batch_process_entries.py` for a prefix/day. These scripts:
     - Pull the event from Mongo.
     - Download burst images from GCS (using `visual_embeddings/gcs.py` + service-account creds).
     - Preprocess the crops to 128x256 and call the ReID embedder (HTTP) to get per-image vectors.
     - Call the VLM service (Vertex AI) to produce attribute labels/confidence in the schema expected by `/match`.
     - Emit a JSON payload ready for `/match` with fields: `day_id`, `store_id`, `entry_id`, `embeddings`, `attrs`, `topk`, `persist`.
     - Optionally log whether that `entry_id` already exists in the matcher’s Mongo persistence (`entries` collection).

3. **Posting to matcher.**
   - Use `entry_pipeline/post_payloads.py` (or curl manually) to POST each payload to `POST /match`.

4. **Matcher internals (`matcher.py`).**
   - Loads the metric encoder checkpoint (`metric_model.EntryEncoder`), attribute schema, and (optional) Mongo connection.
   - Maintains in-memory FAISS `IndexFlatIP` instances per `(day_id, store_id)` bucket to keep matching scoped to a store-day.
   - On `/match`:
     1. Normalize each incoming per-image embedding and fuse them (plus attrs) with the metric encoder.
     2. Search FAISS for top-K cosine similarities against the bucket.
     3. Apply scoring rule: declare `status: "match"` only if `nn1 >= THRESHOLD` and `(nn1 - nn2) >= MARGIN`; otherwise `status: "new"`. Here `nn1` is the cosine similarity between the incoming embedding and FAISS’s top-1 neighbor, while `nn2` is the top-2 similarity (or `-1` if fewer than two entries exist). Both values are computed by FAISS as the dot product of L2-normalized vectors.
     4. Append the new vector to FAISS regardless of status, so the entry becomes available for downstream matches in the same store/day.
     5. If `persist=true` and `MONGO_URI` is configured, upsert the raw `embeddings` and `attrs` into `db.entries` (keyed by `_id = entry_id`, with `day_id` and `store_id` stored alongside). This supports rebuilds and guard rails for replayed entries.

5. **Persistence & rebuild.**
   - `matcher.rebuild_from_mongo(day_id)` clears the FAISS bucket for that day and repopulates it from Mongo’s `entries` collection, re-encoding each persisted document. Use `POST /rebuild` to trigger it per day when restarting the service.

6. **Storage summary.**
   - **FAISS (in-memory):** Stores the fused embeddings (unit-length float32 vectors) for every entry, per `(day_id, store_id)`. Volatile—cleared on restart unless rebuilt.
   - **Mongo (`entries` collection):** Stores the raw per-image embeddings and the attribute dict for each entry. Acts as the source of truth for rebuilds and allows the tooling to check if an `entry_id` already exists before regenerating payloads.

7. **Matching semantics.**
   - Cosine similarity via FAISS on L2-normalized vectors.
   - `THRESHOLD` and `MARGIN` (default 0.88 / 0.02) govern precision. Adjust via env vars if you need to trade recall vs precision.
   - `topk` field in the payload can override the configured K for that specific match call, but matching is still scoped to the same `(day_id, store_id)` bucket.




# Production Monitoring
Monitor the matching-service logs:
```bash
kubectl -n victra-poc logs -f deployment/matching-service
watch kubectl -n victra-poc top pods
```
Monitor pod deploys:
```bash
watch kubectl -n victra-poc get pods
```
Describe last crash
```bash
kubectl -n victra-poc describe pod manifest-editor
```