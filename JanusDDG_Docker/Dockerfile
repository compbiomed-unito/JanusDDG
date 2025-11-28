# 1. Usa ESATTAMENTE Python 3.11.5 (versione leggera)
FROM python:3.11.5-slim

WORKDIR /app

# 2. Installa git e compilatori (essenziali per alcune librerie matematiche)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Copia il requirements originale
COPY requirements.txt .

# 4. FILTRAGGIO E FIX AUTOMATICO:
# - grep -v: Rimuove torch-geometric (perché userai la tua cartella locale)
# - sed: Corregge l'errore "torch=2.4.1" trasformando il singolo "=" in "=="
RUN grep -v "torch-geometric" requirements.txt | grep -v "torch_geometric" | sed 's/\([^=<>!]\)=\([0-9]\)/\1==\2/g' > requirements_filtered.txt

# 5. STEP FONDAMENTALE: Installiamo PRIMA torch da solo.
# torch-scatter e torch-sparse hanno bisogno di torch GIA' installato per potersi compilare.
# Se li mettiamo tutti insieme nello stesso comando pip, fallisce.
RUN pip install --no-cache-dir torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Installa il resto delle librerie
# Ora che torch c'è, torch-scatter si installerà senza errori.
RUN pip install --no-cache-dir -r requirements_filtered.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 7. Installa Jupyter
RUN pip install notebook jupyterlab

# 8. Copia tutto il resto (il notebook e la tua cartella locale "torch_geometric")
COPY . .

# 9. Imposta PYTHONPATH per usare la tua cartella locale
# Questo permette al codice di fare "import torch_geometric" usando la cartella che hai copiato
ENV PYTHONPATH="/app:${PYTHONPATH}"

# 10. Avvio
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]