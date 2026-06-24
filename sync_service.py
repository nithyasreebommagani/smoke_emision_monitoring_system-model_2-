import time
import requests

from db import get_pending_events, mark_synced

WEBHOOK_URL = "http://127.0.0.1:5000/smoke-event"

while True:

    pending = get_pending_events()

    print(f"Pending Events: {len(pending)}")

    for row in pending:

        event_id = row[0]

        payload = {
            "vehicle_id": row[1],
            "plate": row[2],
            "timestamp": row[3],
            "smoke_count": row[4]
        }

        try:

            response = requests.post(
                WEBHOOK_URL,
                json=payload,
                timeout=5
            )

            if response.status_code == 200:

                mark_synced(event_id)

                print(
                    f"Synced Event {event_id}"
                )

        except Exception as e:

            print(
                f"Failed Event {event_id}: {e}"
            )

    time.sleep(30)