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
          reId:
            matchingServiceURL: http://matching-service
            visualAttrPromptName: reid-visual-attributes
            visualAttrResponseModel: reid_visual_attribute_classifier
            visualAttrModelName: projects/815276040543/locations/us-central1/endpoints/1385445124137287680

EOF