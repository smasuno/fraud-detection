FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Vertex AI sends requests to port 8080
ENV PORT=8080
ENV MODEL_PATH=artifacts/model.bst
ENV PREPROCESSOR_PATH=artifacts/preprocessor.pkl
ENV FEATURE_NAMES_PATH=artifacts/feature_names.json
ENV THRESHOLD=0.5

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
