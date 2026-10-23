FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Tehran PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# PyPI is unreliable from Iranian datacenters; Liara's mirror is not
ARG PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt
COPY pishro ./pishro
COPY config ./config
COPY models ./models
COPY reports ./reports
EXPOSE 8000
CMD ["python", "-m", "pishro", "serve", "--port", "8000"]
