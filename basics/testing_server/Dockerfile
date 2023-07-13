FROM continuumio/anaconda3
MAINTAINER "Sachin Rajan" 
COPY ./tf_serve_trial /usr/local/python
EXPOSE 5000
WORKDIR /usr/local/python
RUN pip install -r requirements.txt
CMD python flask_predict_api.py
