#!/usr/bin/env bash
set -eo pipefail

echo "Terminating any stale processes from previous runs..."
pkill -f "uvicorn app:app --host 0.0.0.0 --port 8884" || true
pkill -f "uvicorn manifest-editor.manifest_editor_server:app" || true
pkill -f "npm run dev" || true
pkill -f "kubectl.*port-forward" || true
if [ -d "/tmp/reid" ]; then
    rm -f /tmp/reid/{*.pid,*.log}
fi
echo "Stale process cleanup complete."

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
    TARGET="victra-poc"
fi

export NAMESPACE=${TARGET}
export CONTEXT="gke_teknoir_us-central1-c_teknoir-cluster"
export PROJECT="teknoir"
export CLUSTER="teknoir-cluster"
export DOMAIN="teknoir.cloud"
mkdir -p /tmp/reid

context_exists() {
    kubectl config get-contexts -o name | grep -q "^$1$"
}

check_port() {
    local port=$1
    if nc -z localhost $port 2>/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to check if a specific service is already running
check_service_running() {
    local service=$1
    local port=$2
    local pidfile="/tmp/reid/$service.pid"

    # Check if pidfile exists and process is still running
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$service] Already running (PID: $pid)"
            return 0
        else
            echo "[$service] Stale pidfile found, removing..."
            rm "$pidfile"
            return 1
        fi
    fi

    # Check if port is in use (might be running from another terminal)
    if check_port $port; then
        echo "[$service] Port $port is already in use (running from another terminal?)"
        return 0
    fi

    return 1
}

port_forward() {
  echo "kubectl --context=$CONTEXT port-forward -n $4 svc/$1"
  kubectl --context=$CONTEXT port-forward -n "$4" svc/$1 $2:$3 2>&1 &
  child_pid=$!
  echo "$child_pid" > "/tmp/reid/$service.pid"
}

if context_exists "$CONTEXT"; then
  export CONTEXT="$CONTEXT"
else
  export CONTEXT="teknoir"
fi

kubectl config use-context ${CONTEXT}

if [ -z "$TARGET" ]; then
    echo "Error: No target specified. Please specify a target with -t or --target"
    echo "Usage: $0 -t <target>"
    echo "Available targets: demonstrations, victra-poc, teknoir-ai, boxer-property"
    exit 1
fi

echo "Using context: $CONTEXT"
echo "Using namespace: $NAMESPACE"


#export -f port_forward

forward_and_catch() {
  service=$1
  port_from=$2
  port_to=$3
  ns=$4
  echo "[$service] Starting port forwarding"
  has_error=true
  while [[ "${has_error}" == "true" ]]; do
    exec 3< <(port_forward ${service} ${port_from} ${port_to} ${ns})
    has_error=false
    while IFS= read <&3 line && [[ "${has_error}" == "false" ]]
      do
        child_pid=$(cat "/tmp/reid/$service.pid")
        if [[ $line == *"broken pipe"* || $line == *"Timeout"* ]]; then
          echo "[$service] ERROR: $line"
          kill -9 "$child_pid"
          echo "[$service] Restarting port forwarding"
          has_error=true
          break
        else
          echo "[$service][$child_pid] $line"
        fi
      done
  done
  echo "[$service] Port forwarding has stopped"
}

start_local_service() {
  service=$1
  cmd=$2
  logfile="/tmp/reid/$service.log"
  echo "[$service] Starting service (logging to $logfile)"
  eval "$cmd" > "$logfile" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "/tmp/reid/$service.pid"
  tail -f "$logfile" &
}


if ! check_service_running "mongodb" 27017; then
    forward_and_catch "mongodb" 27017 27017 "$NAMESPACE" &
fi

if ! check_service_running "re-id-mongo" 37017; then
    forward_and_catch "re-id-mongo" 37017 27017 "$NAMESPACE" &
fi

if ! check_service_running "sdmc-media-service" 8882; then
    forward_and_catch "sdmc-media-service" 8882 80 "$NAMESPACE" &
fi



echo "Waiting for essential services..."
for i in {1..30}; do
    if nc -z localhost 27017 2>/dev/null && nc -z localhost 37017 2>/dev/null && nc -z localhost 8882 2>/dev/null; then
        echo "Essential services are ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Timeout waiting for services"
        exit 1
    fi
    sleep 1
done

cleanup() {
    echo "Cleaning up port forwarding processes..."
    if [ -d "/tmp/reid" ]; then
        for pidfile in /tmp/reid/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                kill -9 "$pid" 2>/dev/null || true
                rm "$pidfile"
            fi
        done
    fi
}

trap cleanup EXIT

# Set environment variables for local development
export MONGODB_PASSWORD=$(kubectl --context=$CONTEXT --namespace=$NAMESPACE get secret mongodb-credentials -o yaml | yq .data.password | base64 -d)
export MONGODB_USER=$(kubectl --context=$CONTEXT --namespace=$NAMESPACE get secret mongodb-credentials -o yaml | yq .data.username | base64 -d)
export HISTORIAN_MONGODB_URI="mongodb://${MONGODB_USER}:${MONGODB_PASSWORD}@localhost:27017/historian?authSource=admin&readPreference=primary&appname=LC&ssl=false"
export REID_MONGODB_PASSWORD=$(kubectl --context=$CONTEXT --namespace=$NAMESPACE get secret re-id-mongo -o yaml | yq .data.password | base64 -d)
export REID_MONGODB_USER=$(kubectl --context=$CONTEXT --namespace=$NAMESPACE get secret re-id-mongo -o yaml | yq .data.username | base64 -d)
export REID_MONGODB_URI="mongodb://${REID_MONGODB_USER}:${REID_MONGODB_PASSWORD}@localhost:37017/historian?authSource=admin&readPreference=primary&appname=LC&ssl=false"
export MEDIA_SERVICE_BASE_URL="http://localhost:8882/$NAMESPACE/media-service/api"
export BASE_URL="/"

# Activate Python virtual environment
source .venv/bin/activate

# Start local services if not already running
if ! check_service_running "matching-service" 8884; then
    export BUCKET_PREFIX="gs://${NAMESPACE}.${DOMAIN}"
    export REID_MONGODB_URI=${REID_MONGODB_URI}
    start_local_service "matching-service" "(cd matching-service && MODEL_CKPT='models/encoder/model.pt' uvicorn app:app --host 0.0.0.0 --port 8884)"
fi
if ! check_service_running "manifest-editor" 8883; then
    export MANIFEST_API_BASE=http://localhost:8884
    export MANIFEST_API_TIMEOUT_SECONDS=120
    export MANIFEST_EDITOR_BUCKET="gs://${NAMESPACE}.${DOMAIN}"
    export REID_MONGODB_URI=${REID_MONGODB_URI}
    start_local_service "manifest-editor" "uvicorn manifest-editor.manifest_editor_server:app --host 0.0.0.0 --port 8883 --workers 1"
fi

# Start the app after port forwarding
echo "Starting dev server..."
export PORT=3000
pushd re-identification-system
npm run dev
popd

# Wait for all background processes to complete
echo "All background tasks have completed."


