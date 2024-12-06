import logging
import socket
import sys
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from tenacity import after_log, before_sleep_log, retry, stop_after_attempt, wait_fixed

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
import boto3
from botocore.config import Config

from src.settings import GlobalSettings, get_default_setting
from src.utils import get_formatted_logger

logger = get_formatted_logger(__file__)
# original_getaddrinfo = socket.getaddrinfo

# # Ghi đè hàm getaddrinfo của socket
# socket.getaddrinfo = custom_getaddrinfo


def get_s3_client(
    setting: Annotated[GlobalSettings, Depends(get_default_setting)],
) -> "S3Client":
    return S3Client(
        url=setting.s3_config.endpoint_url,
        service_name=setting.s3_config.service_name,
        region_name=setting.s3_config.region_name,
        access_key=setting.s3_config.access_key,
        secret_key=setting.s3_config.secret_key,
        proxy=setting.s3_config.proxy,
        source_name=setting.s3_config.source_name,
    )


class S3Client:
    """
    S3 client to interact with S3 server
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_fixed(4),
        after=after_log(logger, logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def __init__(
        self,
        url: str,
        access_key: str,
        secret_key: str,
        service_name: str,
        region_name: str,
        source_name: str,
        proxy: dict = None,
    ):
        """
        Initialize S3 client

        Args:
            url (str): S3 url
            access_key (str): S3 access key
            secret_key (str): S3 secret key
        """

        self.source_name = source_name

        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        session_config = Config(
            proxies=proxy,
            signature_version="s3v4",
        )

        self.client = session.client(
            service_name=service_name,
            region_name=region_name,
            endpoint_url=url,
            config=session_config,
        )
        self.test_connection()
        logger.info("S3Client initialized successfully !!!")

    @classmethod
    def from_setting(cls, setting: GlobalSettings) -> "S3Client":
        return cls(
            url=setting.s3_config.endpoint_url,
            access_key=setting.s3_config.access_key,
            secret_key=setting.s3_config.secret_key,
            service_name=setting.s3_config.service_name,
            region_name=setting.s3_config.region_name,
            proxy=setting.s3_config.proxy,
            source_name=setting.s3_config.source_name,
        )

    def test_connection(self):
        """
        Test the connection with the S3 server by listing buckets
        """
        try:
            self.client.list_buckets()
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            raise ConnectionError("S3 connection failed")

    def check_bucket_exists(self, bucket_name: str, check_folder: bool = True) -> bool:
        """
        Check if bucket exists in S3

        Args:
            bucket_name (str): Bucket name

        Returns:
            bool: True if bucket exists, False otherwise
        """
        buckets = self.client.list_buckets()
        for bucket in buckets["Buckets"]:
            objects = self.client.list_objects_v2(Bucket=bucket["Name"])
            if "Contents" in objects:
                for obj in objects["Contents"]:
                    if check_folder and obj["Key"].startswith(bucket_name):
                        return True

                    if not check_folder and obj["Key"] == bucket_name:
                        return True

        return False

    def create_bucket(self, bucket_name: str) -> None:
        """
        Create bucket in S3

        Args:
            bucket_name (str): Bucket name
        """

        try:
            # Tải lên một đối tượng "trống" để tạo thư mục
            self.client.put_object(Bucket=self.source_name, Key=bucket_name)
            print(
                f"Thư mục '{bucket_name}' đã được tạo thành công trong bucket '{bucket_name}'"
            )
        except Exception as e:
            print("Lỗi khi tạo thư mục:", e)
        self.client.make_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} created successfully !!!")

    @retry(stop=stop_after_attempt(3))
    def upload_file(
        self, bucket_name: str, object_name: str, file_path: str | Path
    ) -> None:
        """
        Upload file to S3

        Args:
            bucket_name (str): Bucket name
            object_name (str): Object name to save in S3
            file_path (str | Path): Local file path to be uploaded
        """
        file_path = str(file_path)

        # if not self.check_bucket_exists(bucket_name):
        #     logger.debug(f"Bucket {bucket_name} does not exist. Creating bucket...")
        #     print(f"Bucket {bucket_name} does not exist. Creating bucket...")
        #     self.create_bucket(bucket_name)

        object_name = f"{bucket_name}/{object_name}"

        self.client.upload_file(file_path, self.source_name, object_name)

        if self.check_bucket_exists(object_name, check_folder=False):
            logger.info(f"Uploaded: {file_path} --> {object_name}")
        else:
            logger.info(f"Can't Upload file {object_name}")


    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        """
        Download file from Minio

        Args:
            bucket_name (str): Bucket name
            object_name (str): Object name to download
            file_path (str): File path to save
        """
        # if not self.check_bucket_exists(bucket_name):
        #     logger.warning(f"Bucket {bucket_name} does not exist. Do nothing ...")
        #     return
        object_name = f"{bucket_name}/{object_name}"
        logger.info(f"Downloading: {object_name} --> {file_path}")
        self.client.download_file(self.source_name, object_name, file_path)
        logger.info(f"Downloaded: {object_name} --> {file_path}")

    def remove_file(self, bucket_name: str, object_name: str) -> None:
        """
        Remove file from Minio

        Args:
            bucket_name (str): Bucket name
            object_name (str): Object name to remove
        """
        if not self.check_bucket_exists(bucket_name):
            logger.warning(f"Bucket {bucket_name} does not exist. Do nothing ...")
            return

        object_name = f"{bucket_name}/{object_name}"
        self.client.delete_object(Bucket=self.source_name, Key=object_name)
        logger.debug(f"Removed from S3: {object_name}")
