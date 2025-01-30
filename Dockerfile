FROM python:3.12 
WORKDIR .
COPY test.py .
CMD ["python", "test.py"]