FROM continuumio/miniconda3:latest

WORKDIR /app

# Instalar todo via conda-forge (binarios precompilados, sin compilar C/Fortran)
RUN conda install -y -c conda-forge \
    aurorafusion \
    streamlit \
    plotly \
    scipy \
    numpy \
    && conda clean -afy

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
