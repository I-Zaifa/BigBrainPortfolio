FROM python:3.12-slim
RUN apt-get update && apt-get install -y build-essential libffi-dev
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
EXPOSE 8501
COPY . /app
WORKDIR /app
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true"]
