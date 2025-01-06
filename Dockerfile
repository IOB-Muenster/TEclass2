FROM continuumio/miniconda3

WORKDIR /app

# Setup conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Copy files
COPY TEclass2.py .
COPY data/vocabs/ data/vocabs/
COPY utils/ utils/
COPY dnaformer/ dnaformer/

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Command to run the Python script
ENTRYPOINT ["./entrypoint.sh"]