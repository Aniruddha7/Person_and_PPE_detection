# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt to the container
COPY requirements.txt .

# Install dependencies with caching optimization
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --default-timeout=1000 -i https://pypi.org/simple -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 5000

# Run the application
CMD ["python", "person_ppe_inference/app.py"]
