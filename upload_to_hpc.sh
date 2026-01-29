#!/bin/zsh

# Upload script using rsync to transfer data to TAMU Grace cluster
# Syncs .venv, .vscode, scripts, src folders and all root-level files
# Author: Generated script
# Date: 2026-01-28

# Configuration
REMOTE_USER="prasad.bandodkar"
REMOTE_HOST="grace.tamu.edu"
REMOTE_LOGIN="${REMOTE_USER}@${REMOTE_HOST}"

# Path to configuration file
CONFIG_FILE="paths.txt"

# Color output (defined early for error messages)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo "${RED}[ERROR]${NC} $1"
}

# Check if paths.txt exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    print_error "Configuration file not found: ${CONFIG_FILE}"
    echo ""
    echo "Expected format in ${CONFIG_FILE}:"
    echo "  local_path=/path/to/local/project"
    echo "  remote_path=/path/on/remote/cluster"
    exit 1
fi

# Read paths from configuration file
LOCAL_PATH=""
REMOTE_PATH=""

while IFS='=' read -r key value || [[ -n "$key" ]]; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    
    # Trim whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    case "$key" in
        local_path)
            LOCAL_PATH="$value"
            ;;
        remote_path)
            REMOTE_PATH="$value"
            ;;
    esac
done < "${CONFIG_FILE}"

# Validate that both paths were found
if [[ -z "${LOCAL_PATH}" ]]; then
    print_error "local_path not found in ${CONFIG_FILE}"
    exit 1
fi

if [[ -z "${REMOTE_PATH}" ]]; then
    print_error "remote_path not found in ${CONFIG_FILE}"
    exit 1
fi

# Ensure local path ends with trailing slash for rsync
if [[ "${LOCAL_PATH}" != */ ]]; then
    LOCAL_PATH="${LOCAL_PATH}/"
fi

# Rsync options
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose output
# -z: compress data during transfer
# --delete: delete files in remote that don't exist in local (for exact sync)
RSYNC_OPTIONS="-avz"

# Display configuration
echo "================================="
echo "  RSYNC UPLOAD TO GRACE CLUSTER  "
echo "================================="
echo ""
print_info "Configuration:"
echo "  Local path:  ${LOCAL_PATH}"
echo "  Remote host: ${REMOTE_HOST}"
echo "  Remote path: ${REMOTE_LOGIN}:${REMOTE_PATH}"
echo ""
print_info "Uploading entire directory"
echo ""

# Check if local path exists
if [[ ! -d "${LOCAL_PATH}" ]]; then
    print_error "Local path does not exist: ${LOCAL_PATH}"
    echo ""
    echo "Please verify the local_path in ${CONFIG_FILE}"
    exit 1
fi

# Confirm before proceeding
print_warning "This will upload data to the cluster and replace existing files."
echo -n "Do you want to continue? [y/N]: "
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    print_info "Upload cancelled."
    exit 0
fi

echo ""
print_info "Starting rsync transfer..."
echo ""

# Perform the rsync
rsync ${RSYNC_OPTIONS} "${LOCAL_PATH}" "${REMOTE_LOGIN}:${REMOTE_PATH}"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    print_info "Upload completed successfully!"
else
    echo ""
    print_error "Upload failed with exit code $?"
    exit 1
fi

echo ""
print_info "Done!"
