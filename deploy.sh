#!/usr/bin/env bash
set -e
#set -x

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -t|--target)
    TARGET="$2"
    shift
    shift
    ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
esac
done

if [ -z "$TARGET" ]; then
    # Default target if none provided
    TARGET="victra-poc"
fi

# Build the Docker images first
export BRANCH_NAME=${BRANCH_NAME:-"local-build"}
export SHORT_SHA=${SHORT_SHA:-$(date +%Y%m%d-%H%M%S)}
pushd re-identification-system
./build.sh
popd
pushd manifest-editor
./build.sh
popd
pushd matching-service
./build.sh
popd

export NAMESPACE=${TARGET}
if [[ $NAMESPACE == "demonstrations" ]] ; then
  CONTEXT="gke_teknoir-poc_us-central1-c_teknoir-dev-cluster"
  DOMAIN="teknoir.dev"
else
  CONTEXT="gke_teknoir_us-central1-c_teknoir-cluster"
  DOMAIN="teknoir.cloud"
fi

cat <<EOF | kubectl --context "$CONTEXT" --namespace "$NAMESPACE" apply -f -
---
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: re-identification-system
  namespace: ${NAMESPACE}
spec:
  repo: https://teknoir.github.io/re-identification-system
  chart: re-identification-system
  version: 0.0.10
  targetNamespace: ${NAMESPACE}
  valuesContent: |-
    basePath: /${NAMESPACE}/re-identification-system
    manifestApiBaseUrl: http://matching-service
    domain: ${DOMAIN}
    mediaServiceBaseUrl: https://${DOMAIN}/${NAMESPACE}/media-service/api
    image:
      tag: ${BRANCH_NAME}-${SHORT_SHA}
    matchingService:
      modelCheckpoint: /app/matching-service/models/encoder/model.pt
      image:
        tag: ${BRANCH_NAME}-${SHORT_SHA}
    manifestEditor:
      image:
        tag: ${BRANCH_NAME}-${SHORT_SHA}

    triton:
      models:
        - name: swin-reid
          image: us-docker.pkg.dev/teknoir/gcr.io/swin-reid-triton:latest-local-20251112-104202

    event-processing-pipeline:
      streams:
        - stream: cloud-line-crossing
          #debugLevel: DEBUG
          domain: ${DOMAIN}
          image:
            repository: us-docker.pkg.dev/teknoir/gcr.io/observatory-event-processing
            tag: feature-line-crossing-cloud-stream-25aebe2
          serviceAccountName: default-editor
          instructor:
            model: projects/815276040543/locations/us-central1/endpoints/1385445124137287680
            #model: gemini-2.5-flash
            prompt: |
              You are a vision assistant that extracts a compact set of clothing
              attributes used for re-ID of persons across multiple camera angles.
              Return STRICT JSON ONLY (no prose) including every schema key exactly:
              one_piece, outerwear, top_wear, bottom_wear, footwear, footwear_color,
              head_covering, skin_tone,
              outerwear_color, top_wear_color, bottom_wear_color, head_covering_color,
              one_piece_color,
              outerwear_pattern, top_wear_pattern, bottom_wear_pattern,
              head_covering_pattern, one_piece_pattern,
              and *_confidence per key as floats in [0,1]. Do not omit keys.
              
              Values must be lowercase and from these vocabularies:
              one_piece: dress|jumpsuit|romper|coverall|none|unknown
              outerwear: jacket|coat|hoodie|vest|sweater|none|unknown
              top_wear: t-shirt|shirt|blouse|sweater|hoodie|polo|none|unknown
              bottom_wear: pants|jeans|shorts|skirt|none|unknown
              footwear: athletic shoes|boots|dress shoes|sandals|unknown
              footwear_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              outerwear_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              top_wear_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              bottom_wear_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              head_covering_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              one_piece_color: black|white|gray|brown|red|blue|green|yellow|multicolor|unknown
              head_covering: cap|beanie|hat|hoodie|none|unknown
              skin_tone: light|medium|dark|unknown
              outerwear_pattern: plain|patterned|logo_writing|unknown
              top_wear_pattern: plain|patterned|logo_writing|unknown
              bottom_wear_pattern: plain|patterned|logo_writing|unknown
              head_covering_pattern: plain|patterned|logo_writing|unknown
              one_piece_pattern: plain|patterned|logo_writing|unknown
            
              Notes:

              * Multi-frame rule: Only use 'unknown' if NO image shows the attribute. If at
              least one image shows it, choose the majority value; if tie, prefer the more
              specific category (e.g., beanie over hat). For *_color, if colors conflict
              across frames, use 'multicolor' instead of 'unknown'.

              * Badge/Lanyard rule: Employees frequently wear badges or lanyards. If they are
              wearing one over a plain shirt, make sure to mark the top_wear_pattern as
              'plain'. It's also common for employees to wear a badge or lanyard over a tshirt
              with a logo. In that instance, mark the top_wear_pattern as 'logo_writing'.
              
              * Head covering rule
              ** If a person is wearing a durag, mark head_covering as 'beanie'
              ** If a person is wearing a golf cap, mark head_covering as 'beanie'

              * Delivery person rule: UPS drivers frequently wear brown uniforms where it looks
              like they are wearing a one_piece 'coverall'. These are not coveralls but
              uniforms with matching brown outerwear 'shirt' and bottom_wear 'pants' or
              'shorts'.

              * Color mapping rules:
              ** pink, peach, or orange -> 'red'
              ** purple -> 'blue'

              * Shoe rules:
              **If a shoe appears to mainly be one color with a clearly different colored sole,
              use 'multicolor'
              **If a person is wearing Crocs (closed toe with lots of holes and a strap at the
              back) mark footwear as 'sandals'.

              * Top-wear rules:
              **If a top_wear is short sleeve and has buttons below the chest, mark it as a
              'shirt' instead of a 'polo'.
              **If the top_wear appears to be a 't-shirt' but has a distinguishable collar,
              mark it as a 'polo'.
              **Striped top_wear should be marked as 'patterned'.
              **If a shirt has a face on it, mark it as 'logo_writing'.
              **Shirts with a tiny logo on a sleeve should be marked as 'plain' (they may not
              be visible from multiple camera angles).
              ** Sweatshirts without a hood should be marked as 'sweater'.
              ** If a person is wearing a hoodie or sweater with no other visible garment, mark top_wear as 'hoodie' or 'sweater', leave outerwear as 'none'.
              ** If a shirt has an image or character on it, mark it as 'logo_writing'.

              * Other
              ** Ripped jeans rule: If jeans are ripped, mark them as 'patterned'.
              ** If a woman is wearing a burqa, mark one_piece as 'dress', 'head_covering' as
              'unknown', 'top_wear' as 'none', 'bottom_wear' as 'none'.
              ** If a person is wearing leggings, mark bottom_wear as 'pants'.
              ** If a person is wearing a tracksuit, mark top_wear as 'sweater' and bottom_wear
              as 'pants'.
              ** If an article of clothing is both patterned and has a logo, mark it as
              'patterned'.

              Confidence Score Calibration:
              - The confidence score should reflect the certainty of the assigned attribute LABEL.
              - A confidence of 1.0 should only be used when the attribute is clearly and unambiguously visible across multiple high-quality images.
              - If an attribute is visible in only one image, is partially obscured, the confidence should be lowered (e.g., 0.7-0.9).
              - If an attribute is inferred but not clearly seen (e.g., assuming 'pants' when only the upper body is visible), the confidence should be significantly lower (e.g., 0.4-0.6).
              - If a label is set to 'unknown' because the attribute is not visible or identifiable in any of the images, this indicates a lack of evidence for a specific label. Therefore, the confidence score should be low (e.g., 0.1-0.3), reflecting the uncertainty in assigning any specific, known attribute.


              Before generating the JSON output, for each attribute, internally consider the following:
              1.  **Evidence**: Which images show this attribute?
              2.  **Clarity**: How clear and visible is the attribute in those images?
              3.  **Consistency**: Is the attribute consistent across all images?
              4.  **Justification**: Based on the evidence, what is the appropriate value and a justifiable confidence score according to the calibration rules?
              
EOF