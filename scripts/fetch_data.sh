#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"

mkdir -p "$RAW_DIR"
touch "$RAW_DIR/.gitkeep"

download() {
  local out="$1"
  shift
  local urls=("$@")

  if [[ -f "$RAW_DIR/$out" ]]; then
    echo "Skipping $out (already exists)"
    return 0
  fi

  for url in "${urls[@]}"; do
    echo "Trying $out from: $url"
    if curl -fL --retry 3 --retry-delay 2 "$url" -o "$RAW_DIR/$out"; then
      echo "Downloaded: $out"
      return 0
    fi
  done

  echo "Failed to download $out automatically."
  echo "Please download manually and place it at: $RAW_DIR/$out"
  return 1
}

download "NIST.CSWP.29.pdf" \
  "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"

download "NIST.SP.800-61r3.pdf" \
  "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf"

download "NIST.SP.1299.pdf" \
  "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1299.pdf"

download "OWASP-Top-10-for-LLMs-v2025.pdf" \
  "https://genai.owasp.org/wp-content/uploads/2025/02/OWASP-Top-10-for-LLMs-v2025.pdf" \
  "https://raw.githubusercontent.com/OWASP/www-project-top-10-for-large-language-model-applications/main/document/OWASP-Top-10-for-LLMs-v2025.pdf"

echo
echo "Done. Files in: $RAW_DIR"
ls -lh "$RAW_DIR"
