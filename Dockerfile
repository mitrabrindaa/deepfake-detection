# Use a standard Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all your files into the container
COPY . /app

# Install system dependencies needed for OpenCV/Albumentations
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Hugging Face expects
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]