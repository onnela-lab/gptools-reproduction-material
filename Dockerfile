FROM python:3.10
WORKDIR /workdir
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m cmdstanpy.install_cmdstan --verbose --version 2.33.0

