FROM continuumio/miniconda3
WORKDIR /app
# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# Conda environment requirements
COPY grasp_gym_dep.yaml /app/grasp_gym_dep.yaml
# Install the environment
RUN conda env create -f /app/grasp_gym_dep.yaml
# Activate the environment
SHELL ["conda", "run", "-n", "gym_env", "/bin/bash", "-c"]
# Copy the rest of the project files
COPY . /app
# Default shell
CMD ["bash"]