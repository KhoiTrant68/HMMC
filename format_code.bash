#!/bin/bash
set -euo pipefail

# Optimize: concise output, robust checks
for tool in isort black; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "❌ Required tool '$tool' not found. Please install it."
        exit 1
    fi
done

echo "🔄 Cleaning Python cache files..."
find . -type d -name "__pycache__" -prune -exec rm -rf '{}' +

echo "🧹 Running isort (sort & remove unused imports)..."
isort .

echo "🎨 Formatting with black..."
black .

# Optimize: add summary
if [ $? -eq 0 ]; then
  echo "✅ Code formatting complete."
fi
