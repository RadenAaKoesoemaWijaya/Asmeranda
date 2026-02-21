#!/bin/bash
# Azure startup script for Streamlit application

echo "Starting Streamlit application on Azure..."

# Set Azure-specific environment variables
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create necessary directories if they don't exist
mkdir -p /app/models /app/uploads /app/interpretation_cache /app/logs

# Set proper permissions
chmod -R 755 /app/models /app/uploads /app/interpretation_cache /app/logs

# Log startup information
echo "Application starting at: $(date)"
echo "Streamlit server port: $STREAMLIT_SERVER_PORT"
echo "Streamlit server address: $STREAMLIT_SERVER_ADDRESS"

# Start Streamlit with optimized settings for Azure
exec streamlit run app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false \
    --logger.level=info