FROM python:3.12-slim
# Set the working directory inside the container
WORKDIR /code
# Copy the requirements file from our folder into / code folder in the container
COPY ./requirements.txt /code/requirements.txt
# Install the requirements.txt file inside the 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /create/app

# Set environment variables
ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""

EXPOSE 8080

# Command to run the application
#CMD ["fastapi", "run", "app/main.py", "--port", "8080"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]