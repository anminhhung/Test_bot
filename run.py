import subprocess

def main():
    cmd_str = "uvicorn app:app --host 0.0.0.0 --port 8000 & celery -A src worker -l INFO" #  & celery -A src worker -l INFO
    subprocess.run(cmd_str, shell=True)

if __name__ == '__main__':
    print("Starting the server")
    main()