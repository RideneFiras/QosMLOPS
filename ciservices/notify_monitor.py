import psutil
import time
import requests
import logging
from plyer import notification

# Thresholds
CPU_THRESHOLD = 80  # percent
MEM_THRESHOLD = 80  # percent

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamic Elasticsearch URL
IS_DOCKER = False  # Set to True if running in Docker
ELASTICSEARCH_URL = (
    "http://elasticsearch:9200" if IS_DOCKER else "http://localhost:9200"
)
INDEX_NAME = "monitor-logs"


def notify(title, message):
    notification.notify(title=title, message=message, timeout=5)  # seconds


def send_to_elasticsearch(cpu, mem):
    data = {
        "cpu_percent": cpu,
        "memory_percent": mem,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        response = requests.post(
            f"{ELASTICSEARCH_URL}/{INDEX_NAME}/_doc",
            json=data,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code not in [200, 201]:
            logger.error(f"Failed to send metrics to Elasticsearch: {response.text}")
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")


while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent

    print(f"🔍 CPU: {cpu:.1f}%, RAM: {mem:.1f}%")

    send_to_elasticsearch(cpu, mem)

    if cpu > CPU_THRESHOLD:
        notify("⚠️ High CPU Usage", f"CPU usage at {cpu}%")

    if mem > MEM_THRESHOLD:
        notify("⚠️ High Memory Usage", f"Memory usage at {mem}%")

    time.sleep(5)
