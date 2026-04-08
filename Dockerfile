FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached)
COPY sepsis_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment package
COPY sepsis_env ./sepsis_env

# Make the package importable
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "sepsis_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
