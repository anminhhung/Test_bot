ARG VERSION=3.12

FROM python:${VERSION:-3.11}

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Cài đặt các gói cần thiết
RUN pip install --no-cache-dir --upgrade pip \
    && apt-get update \
    && apt-get install -y telnet ngrep htop net-tools openssl curl wget iputils-ping git libsqlite3-dev supervisor

RUN pip install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir uvicorn[standard] websockets

RUN mkdir -p /var/log/

# Sao chép file cấu hình supervisord vào đúng thư mục
# COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000
# Run app.py when the container launches
# CMD sh -c "uvicorn app:app --host 0.0.0.0 --port 8000 & celery -A src worker -E --loglevel=info"
# COPY supervisord.conf /etc/supervisord.conf
# CMD ["supervisord", "-c", "/etc/supervisord.conf"]

# Thêm tệp entrypoint.sh vào container
# RUN chmod +x entrypoint.sh
# ENTRYPOINT ["sh", "entrypoint.sh"]

CMD sh -c "python run.py"

