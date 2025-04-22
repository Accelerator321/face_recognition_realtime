# Use an official TensorFlow runtime as a base image
FROM tensorflow/tensorflow:2.18.0


# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies from requirements.txt (make sure you have one)
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Python script
CMD ["python", "face_system.py"]
