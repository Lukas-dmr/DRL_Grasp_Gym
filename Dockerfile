FROM continuumio/miniconda3
WORKDIR /app
# Conda environment requirements
COPY environment.yml /app/environment.yml
# Install the environment
RUN conda env create -f /app/environment.yml
# Activate the environment
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]
# Copy the rest of the project files
COPY . /app
# Default shell
CMD ["bash"]