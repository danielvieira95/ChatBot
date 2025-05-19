FROM python:3.13.2-slim

ENV DEBIAN_FRONTEND=noninteractive

# Atualiza e instala dependências de build + Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    git \
    gcc \
    libssl-dev \
    pkg-config \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Por esta:
COPY requirements.lock.txt .
RUN pip install --upgrade pip && pip install -r requirements.lock.txt
# Copia o restante da aplicação
COPY . .

# Garante pasta de documentos
RUN mkdir -p documentos \
    && mkdir -p /tmp/chroma_db \
    && chmod -R 777 /tmp/chroma_db

# Expõe a porta
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
