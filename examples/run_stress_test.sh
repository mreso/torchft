#!/usr/bin/env bash
# Run the torchft + RDMATransport CIFAR-10 stress test under the
# torchft_dev conda env. Forwards extra args to the python script.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-torchft_dev}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# torchvision is required for CIFAR-10 + ResNet-18.
if ! python -c "import torchvision" >/dev/null 2>&1; then
    echo "torchvision not installed — installing CPU build..."
    pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cpu
fi

cd "$REPO_ROOT"
echo "Running stress test from $(pwd)"
echo "Args: $*"
echo

python -u examples/stress_test_cifar10.py "$@"
rc=$?

echo
if [[ $rc -eq 0 ]]; then
    echo "stress test: SUCCESS"
else
    echo "stress test: FAIL (exit $rc)"
fi
exit $rc
