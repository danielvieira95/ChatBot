# Imagem base com Python 3.12
FROM python:3.12

# Evita prompts interativos e define encoding padrão
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Instala compiladores e dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Cria diretório seguro para o ChromaDB
RUN mkdir -p /tmp/chroma_db && chmod -R 777 /tmp/chroma_db
# Copia dependências e instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Expõe porta do FastAPI
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
