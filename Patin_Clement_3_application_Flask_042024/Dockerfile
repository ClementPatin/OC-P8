FROM python:3.11.8


COPY requirements.txt app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt
COPY . /app

ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=8501
ENV FLASK_DEBUG=1

EXPOSE 8501

ENTRYPOINT ["flask", "run"]

CMD ["--host", "0.0.0.0", "--port", "8501"]