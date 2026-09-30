FROM python:3.11-slim

WORKDIR /code

# Dependencies first: Docker caches this layer until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code and the trained models.
COPY app ./app
COPY models ./models

# Run as a normal user instead of root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
