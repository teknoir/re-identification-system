#!/usr/bin/env bash
set -eo pipefail
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
    TARGET="demonstrations"
fi

export BRANCH_NAME=${BRANCH_NAME:-"local"}
export SHORT_SHA=$(date +%Y%m%d-%H%M%S)
export IMAGE="us-docker.pkg.dev/teknoir/gcr.io/re-identification-system"

docker buildx build \
  --platform=linux/amd64 \
  --push \
  --tag "${IMAGE}:${BRANCH_NAME}-${SHORT_SHA}" \
  .

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
  version: 0.0.2-beta-8
  targetNamespace: ${NAMESPACE}
  valuesContent: |-
    basePath: /${NAMESPACE}/re-identification-system
    domain: ${DOMAIN}
    mediaServiceBaseUrl: https://${DOMAIN}/${NAMESPACE}/media-service/api
    image:
      repository: ${IMAGE}
      tag: ${BRANCH_NAME}-${SHORT_SHA}

    event-processing-pipeline:
      streams:
        - stream: cloud-line-crossing
          #debugLevel: DEBUG
          domain: ${DOMAIN}
          image:
            repository: us-docker.pkg.dev/teknoir/gcr.io/observatory-event-processing
            tag: feature-line-crossing-cloud-stream-6afbeda
          mongodbSecretKeyRef:
            name: re-id-mongo
            key: uri
          serviceAccountName: default-editor
EOF