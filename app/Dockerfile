#base image
FROM python:3.10-slim

#create the working directory in the container
WORKDIR /app

#any command like run copy cmd can be done on that relative path
#it will copy file from local to contaier in the specified path
COPY requirements.txt .
COPY mlflow.db .
COPY UnEmployement_Rate_Prediction.ipynb .
COPY app/. .


#Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#EXPOSE THE PORT
EXPOSE 8000

#Run the Fast API, these commands will executed when the container is start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]