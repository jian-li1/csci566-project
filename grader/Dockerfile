FROM python:3.11-slim
COPY grader/requirements.txt /grader-requirements.txt
RUN pip install --no-cache-dir -r /grader-requirements.txt
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
ENV PYTHONHASHSEED=42 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /tmp
