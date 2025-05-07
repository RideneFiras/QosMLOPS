import psutil
import time
from plyer import notification

# Thresholds
CPU_THRESHOLD = 80  # percent
MEM_THRESHOLD = 80  # percent

from plyer import notification

def notify(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=5  # seconds
    )

while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent

    print(f"🔍 CPU: {cpu:.1f}%, RAM: {mem:.1f}%")

    if cpu > CPU_THRESHOLD:
        notify("⚠️ High CPU Usage", f"CPU usage at {cpu}%")

    if mem > MEM_THRESHOLD:
        notify("⚠️ High Memory Usage", f"Memory usage at {mem}%")

    time.sleep(5)
