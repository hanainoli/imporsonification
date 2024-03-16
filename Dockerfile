# Use the official Python image as the base image
FROM python:3.9-slim

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application files into the container
COPY app.py .

# Expose the port on which your FastAPI application will run
EXPOSE 5000

# Command to run the FastAPI application using Uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
