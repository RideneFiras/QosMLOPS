import subprocess
from plyer import notification

def notify_failure():
    notification.notify(
        title="❌ CI Failed",
        message="Check your code for lint/test errors.",
        timeout=5  # seconds
    )

result = subprocess.run(["make", "ci"])

if result.returncode != 0:
    notify_failure()
