FROM public.ecr.aws/docker/library/python:3.11.2-buster
# Thiết lập biến môi trường
ARG SERVICE_NAME
ARG SONAR_PROJECT_KEY
ARG SONAR_HOST_URL
ARG SONAR_TOKEN

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép các tệp cần thiết
COPY . /app

# Cài đặt các gói cần thiết
RUN pip install --no-cache-dir --upgrade pip \
    && apt-get update \
    && apt-get install -y telnet ngrep htop net-tools openssl curl wget iputils-ping git libsqlite3-dev supervisor

RUN pip install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir uvicorn[standard] websockets

# Thiết lập PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN mkdir -p /var/log/

EXPOSE 8000
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD sh -c "python run.py"
# CMD sh -c "uvicorn app:app --host 0.0.0.0 --port 8000 & celery -A src worker -E --loglevel=info"
# COPY supervisord.conf /etc/supervisord.conf
# CMD ["supervisord", "-c", "/etc/supervisord.conf"]

RUN chmod +x entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]