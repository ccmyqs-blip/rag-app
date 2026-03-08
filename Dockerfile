# 国内服务器可用：从 docker.1ms.run 拉取，避免 Docker Hub 超时
FROM docker.1ms.run/library/python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 国内加速：使用阿里云 Debian 镜像，避免 apt 拉包超时
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true

# Install system dependencies (for building wheels, PDF/image handling, OCR, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        libglib2.0-0 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definitions and install Python dependencies first (better layer caching)
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application code
COPY . .

# Ensure data, chroma_db and logs directories exist (will be bind-mounted in production)
RUN mkdir -p /app/data /app/chroma_db /app/logs

EXPOSE 8501

# Default command: run Streamlit app on 0.0.0.0:8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

