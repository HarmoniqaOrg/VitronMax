FROM python:3.10-slim

WORKDIR /app

# Pre-install build tools needed by rdkit-pypi wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libxrender1 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
