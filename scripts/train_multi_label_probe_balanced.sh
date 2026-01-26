#!/bin/bash
set -euo pipefail

echo "Deprecated: use scripts/train_multi_label_probe_per_sample.sh instead." >&2
exec bash scripts/train_multi_label_probe_per_sample.sh "$@"
