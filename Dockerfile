FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (Aurora puede necesitar compiladores)
RUN apt-get update && apt-get install -y \
    gcc g++ make git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto de Railway
EXPOSE 8501

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
