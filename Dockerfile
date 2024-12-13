FROM python:3.10
# Install R and dependencies.
RUN apt-get update && apt-get install -y \
    r-base \
    r-cran-devtools \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workdir
COPY setup.R .
RUN Rscript setup.R

# Install Python dependencies and compile cmdstan.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m cmdstanpy.install_cmdstan --verbose --version 2.36.0
