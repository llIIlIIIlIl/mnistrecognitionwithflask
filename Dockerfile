FROM python:3.7-slim
COPY requirements.txt /app/
COPY main.py /app
WORKDIR /app
RUN pip install -r ./requirements.txt
EXPOSE 3000
CMD ["python","./main.py"]