# Re-Identification System Helm Chart

This chart deploys Re-Identification System application to a Kubernetes cluster.

> The implementation of the Helm chart is right now the bare minimum to get it to work.
> The purpose of the chart is not to be infinitely configurable, but to provide a limited set of configuration options that make sense for the Teknoir platform.

# Helm usage

## Usage in Teknoir platform
Use the HelmChart to deploy the Re-Identification System application to a Namespace.

```yaml
---
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: re-identification-system
  namespace: demonstrations # or any other namespace
spec:
  repo: https://teknoir.github.io/re-identification-system
  chart: re-identification-system
  targetNamespace: demonstrations # or any other namespace
  valuesContent: |-
    # Example for minimal configuration
    
```

## Adding the repository

```bash
helm repo add teknoir-re-identification-system https://teknoir.github.io/re-identification-system/
```

## Installing the chart

```bash
helm install teknoir-re-identification-system teknoir-re-identification-system/re-identification-system -f values.yaml
```
