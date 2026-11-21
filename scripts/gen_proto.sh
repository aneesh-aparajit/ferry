#!/usr/bin/env bash
# Regenerate Python gRPC stubs from proto/ into src/ (module ferry.v1).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --pyi_out=src \
  --grpc_python_out=src \
  proto/ferry/v1/ferry.proto
touch src/ferry/v1/__init__.py
echo "generated src/ferry/v1/ferry_pb2*.py"
