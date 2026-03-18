#!/bin/bash
# Run exercise1 and capture terminal output for portfolio.
# Requires: ollama running with llava pulled, and a test image.
# Usage: ./run_exercise1_test.sh [path_to_image]
# Or: python exercise1_vlm_langgraph_chat_agent.py --cli 2>&1 | tee outputs/exercise1_terminal_output.txt

cd "$(dirname "$0")"
mkdir -p outputs

# Create a minimal test image if none provided
IMG="${1:-}"
if [ -z "$IMG" ] || [ ! -f "$IMG" ]; then
    echo "Creating a minimal test image (red 100x100 pixel)..."
    python3 -c "
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')
img.save('outputs/test_image.jpg')
print('Saved outputs/test_image.jpg')
"
    IMG="outputs/test_image.jpg"
fi

echo "Using image: $IMG"
echo "Running VLM agent (CLI mode). Type 'quit' to exit."
echo ""

# Run with pre-seeded inputs: image path, one question, then quit
# Use python3 from PATH; ensure ollama + llava are available: ollama pull llava
printf "%s\nWhat color is this image?\nquit\n" "$IMG" | python3 exercise1_vlm_langgraph_chat_agent.py --cli 2>&1 | tee outputs/exercise1_terminal_output.txt

echo ""
echo "Output saved to outputs/exercise1_terminal_output.txt"
