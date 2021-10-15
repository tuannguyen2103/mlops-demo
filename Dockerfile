FROM python:3.6
RUN python3.6 -m pip --proxy http://10.61.11.42:3128 install pip --upgrade
WORKDIR /app
COPY ./requirements.txt /app
RUN pip3 --proxy http://10.61.11.42:3128 install -r requirements.txt
COPY . /app
