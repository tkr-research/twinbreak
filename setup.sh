#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 --hf-token <HUGGING_FACE_TOKEN> --store-model-disk-path <STORE_MODEL_DISK_PATH>"
  echo ""
  echo "Example:"
  echo "  $0 --hf-token hf_abc123... --store-model-disk-path /path/to/hf/cache"
  return 1
}

# Parse named parameters
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --hf-token)
      HF_TOKEN="$2"
      shift
      shift
      ;;
    --store-model-disk-path)
      STORE_MODEL_DISK_PATH="$2"
      shift
      shift
      ;;
    --help|-h)
      usage
      return 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      return 1
      ;;
  esac
done

# Validate parameters
if [ -z "$HF_TOKEN" ] || [ -z "$STORE_MODEL_DISK_PATH" ]; then
  echo "Error: both --hf-token and --store-model-disk-path must be provided."
  usage
  return 1
fi

# Get absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy sample.env to .env
cp "$SCRIPT_DIR/sample.env" "$SCRIPT_DIR/.env"

# Replace placeholders in .env
sed -i.bak "s|HF_TOKEN=hf_xxxYOURTOKENxxx|HF_TOKEN=$HF_TOKEN|" "$SCRIPT_DIR/.env"
sed -i.bak "s|STORE_MODEL_DISK_PATH=xxxPATHTOHFCACHExxx|STORE_MODEL_DISK_PATH=$STORE_MODEL_DISK_PATH|" "$SCRIPT_DIR/.env"
rm "$SCRIPT_DIR/.env.bak"

# Print confirmation
echo ".env file created and values injected:"
echo "  - HF_TOKEN"
echo "  - STORE_MODEL_DISK_PATH"

# Export PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR"
echo "PYTHONPATH set to $PYTHONPATH"
