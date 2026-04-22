#!/bin/bash
# Safe restart script for the Mesop app.
# Ensures any previous instance is fully killed and GPUs are clean before starting.

set -e
cd "$(dirname "$0")"

PORT="${PORT:-7861}"

echo "=== Killing previous Mesop process ==="
pids=$(pgrep -f "mesop app_mesop.py" || true)
if [ -n "$pids" ]; then
    echo "Found PIDs: $pids"
    kill $pids 2>/dev/null || true
fi
sleep 2
remaining=$(pgrep -f "mesop app_mesop.py" || true)
if [ -n "$remaining" ]; then
    echo "Force killing stubborn PIDs: $remaining"
    kill -9 $remaining 2>/dev/null || true
fi
sleep 2

# Wait for GPU memory to be released
echo "=== Waiting for GPU cleanup ==="
for i in 1 2 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$used" -lt "1000" ]; then
        echo "GPUs clean (${used} MB used)"
        break
    fi
    echo "GPU still has ${used} MB — waiting ($i/5)..."
    sleep 2
done

# Also clean up any zombie/defunct processes by waiting
echo "=== Starting Mesop on port $PORT ==="
source venv/bin/activate
exec env MESOP_HOST=0.0.0.0 PYTHONUNBUFFERED=1 mesop app_mesop.py --port=$PORT
