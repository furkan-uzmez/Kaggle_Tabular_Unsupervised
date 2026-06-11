#!/usr/bin/env bash
# Download Kaggle TPS Jul 2022 competition data.
# Preconditions:
#   1. `kaggle` CLI installed on host (e.g., `pipx install kaggle`)
#   2. `~/.kaggle/kaggle.json` exists with mode 600
#   3. Competition rules accepted in the Kaggle web UI

set -euo pipefail

COMPETITION="tabular-playground-series-jul-2022"
TARGET_DIR="data/raw"

# Check kaggle CLI is available
if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: 'kaggle' CLI not found on PATH." >&2
    echo "Install with: pipx install kaggle  (or: pip install --user kaggle)" >&2
    exit 1
fi

# Check token exists
if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "ERROR: ${HOME}/.kaggle/kaggle.json not found." >&2
    echo "Create a Kaggle API token at https://www.kaggle.com/settings/account" >&2
    echo "and place it at ~/.kaggle/kaggle.json with mode 600." >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

echo "Downloading ${COMPETITION} into ${TARGET_DIR}/ ..."
kaggle competitions download -c "${COMPETITION}" -p "${TARGET_DIR}"

ZIP_PATH="${TARGET_DIR}/${COMPETITION}.zip"
if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "ERROR: expected ${ZIP_PATH} after download, not found." >&2
    echo "If you got a 403, accept the competition rules in the Kaggle web UI first." >&2
    exit 1
fi

echo "Extracting ..."
(cd "${TARGET_DIR}" && unzip -o "$(basename "${ZIP_PATH}")" && rm -f "$(basename "${ZIP_PATH}")")

echo "Done. Files in ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"
