# --- STAGE 1: Build Stage (Optimized for small image size) ---
# Use a Python base image with a smaller footprint (slim)
# We use the specific version of Python that supports your dependencies
FROM python:3.10-slim

# Set environment variables for better Python and server operation
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages (like Pillow, if needed)
# Although Pillow is usually built, these are good for general stability
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file and install Python dependencies first
# This improves layer caching: if code changes but requirements don't, this layer is reused.
COPY requirements.txt .

# Install dependencies.
# NOTE: Using a PyTorch build optimized for CPU is often preferred in 
# general purpose deployments unless you have a GPU runtime available.
# Replace 'torch==x.x.x' with the specific CPU version if required.
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Final Run Stage ---
# Copy all application files into the container
# This includes all your model architecture and script files:
COPY app.py .
COPY classifier_model.py .
COPY vaegan_model.py .
COPY transform_utils.py .

# Copy the artifacts (model weights)
# This is critical! The artifacts folder must be present.
COPY artifacts/ ./artifacts/

# Expose the port the application will run on
EXPOSE 8000

# Command to run the application using Gunicorn, which is the standard production server.
# Gunicorn settings:
# - workers: Use 2-4 workers (adjust based on CPU core count, 2 is a safe start).
# - bind: Binds to port 8000.
# - app:app: Specifies the entry point (the 'app' object inside 'app.py').
# - timeout: Increase timeout to 120 seconds for slow model loading or inference.
# - worker-class: Use 'uvicorn.workers.UvicornWorker' to handle FastAPI's async nature.
CMD ["gunicorn", "app:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]