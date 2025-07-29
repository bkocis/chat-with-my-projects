# Use Python 3.11 slim image as base
FROM python:3.11-slim

ARG MCP_SERVER_URL

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security first
RUN useradd --create-home --shell /bin/bash streamlituser

# Copy application files
COPY . .

# Create Streamlit config directory and copy config
RUN mkdir -p /home/streamlituser/.streamlit
COPY .streamlit/config.toml /home/streamlituser/.streamlit/

# Change ownership of all files
RUN chown -R streamlituser:streamlituser /app && \
    chown -R streamlituser:streamlituser /home/streamlituser/.streamlit

USER streamlituser

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV MCP_SERVER_URL=${MCP_SERVER_URL}


# Health check for Streamlit app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit with additional flags to prevent interactive prompts
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"] 