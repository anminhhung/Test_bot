import os
import requests
from uuid import UUID
import base64
from dotenv import load_dotenv
from traceback import print_exc
import time 
from .logger import get_formatted_logger

load_dotenv()

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

PROXIES = {
    "http": "",
    "https": ""
}

logger = get_formatted_logger(__file__)


def get_cost_from_trace_id(trace_id: str | UUID) -> float:
    """
    Get cost from trace ID from Langfuse

    Args:
        trace_id (str | UUID): The trace ID

    Returns:
        float: The total cost of the trace ID provided
    """
    trace_id = str(trace_id)

    try:
        
        url = f"{LANGFUSE_HOST}/api/public/traces/{trace_id}"
        response = requests.get(url, auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY), proxies=PROXIES)

        if response.status_code != 200:
            logger.error(
                f"Failed to get cost for trace ID: {trace_id}, status code: {response.status_code}, response: {response.text}. So returning 0.0"
            )
            return 0.0

        data = response.json()

        if "totalCost" not in data:
            logger.warning(
                f"Failed to get cost for trace ID: {trace_id}, response: {response.text}. So returning 0.0"
            )
            return 0.0

        return data["totalCost"]
    except Exception as e:
        print_exc()
        logger.error(f"Failed to get cost for trace ID: {trace_id}. So returning 0.0")
        return 0.0


def get_cost_from_session_id(session_id: str | UUID) -> float:
    """
    Get cost from session ID from Langfuse

    Args:
        session_id (str | UUID): The session ID (conversations.id)

    Returns:
        float: The total cost of all traces in the session
    """
    session_id = str(session_id)

    logger.debug(f"Getting cost for session ID: {session_id}")

    try:
        url = f"{LANGFUSE_HOST}/api/public/sessions/{session_id}"
        logger.info(f"Proxies: {PROXIES}")
        auth = f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}"

        payload = ""
        headers = {"Authorization": f"Basic {base64.b64encode(auth.encode('utf-8')).decode('utf-8')}"}

        response = requests.request("GET", url, data=payload, headers=headers, proxies=PROXIES)

        logger.info(f"URL: {url}")
        logger.info(f"Header: {headers}")
        logger.info(f"Response: {response.text}")

        if response.status_code != 200:
            logger.warning(
                f"Failed to get cost for session ID: {session_id}, status code: {response.status_code}, response: {response.text}. So returning 0.0"
            )
            return 0.0

        data = response.json()

        if len(data["traces"]) == 0:
            logger.debug(f"No traces found for session ID: {session_id}. So returning 0.0")
            return 0.0

        return sum([get_cost_from_trace_id(item["id"]) for item in data["traces"]])
    except Exception as e:
        print_exc()
        logger.error(f"Failed to get cost for session ID: {session_id}. So returning 0.0")
        return 0.0
