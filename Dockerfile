FROM python:3.6
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3.6 -m pip --proxy http://10.61.11.42:3128 install pip --upgrade
WORKDIR /app
COPY ./requirements.txt /app
RUN pip3 --proxy http://10.61.11.42:3128 install -r requirements.txt
COPY . /app
EXPOSE 5000
ENV MODEL_NAME Model
ENV API_TYPE REST 
ENV SERVICE_TYPE MODEL
ENV PERSISTANCE 0

CMD exec seldon-core-microservice ${MODEL_NAME} ${API_TYPE} --service-type ${SERVICE_TYPE}
