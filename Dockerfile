# Dockerfile for CAC Walking Tracker
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs backups

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=basic_app.py
ENV FLASK_ENV=production

# Expose ports
EXPOSE 8000 8501 8765

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting CAC Walking Tracker..."\n\
python3 unified_api.py &\n\
python3 basic_app.py &\n\
streamlit run Streamlit.py --server.port=8501 --server.address=0.0.0.0 &\n\
wait' > start.sh && chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["./start.sh"]
