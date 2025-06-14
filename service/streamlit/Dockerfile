FROM python:3.12

RUN mkdir /streamlit

COPY . /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8501

CMD ["streamlit", "run", "run_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
