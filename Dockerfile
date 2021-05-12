FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install uvicorn==0.13.3
RUN pip3 install pydantic==1.8.1
RUN pip install fastapi==0.65.0
RUN pip install python-multipart==0.0.5

COPY ./model /model/

COPY ./app /app

EXPOSE 5000

CMD ["uvicorn", "--host=0.0.0.0", "--port=5000", "main:app"]
