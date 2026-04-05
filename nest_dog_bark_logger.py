import base64
import json
import logging
import os

import gspread
import requests
from flask import Flask, request
from google.cloud import secretmanager
from google.oauth2.credentials import Credentials

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nest_bark_logger")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DEVICE_ACCESS_PROJECT_ID = os.environ.get("DEVICE_ACCESS_PROJECT_ID")
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
SHEET_NAME = os.environ.get("SHEET_NAME", "Sheet1")

SDM_BASE_URL = "https://smartdevicemanagement.googleapis.com/v1"
TOKEN_URI = "https://oauth2.googleapis.com/token"
SECRET_NAME = "nest-oauth-refresh-token"

SOUND_EVENT_KEY = "sdm.devices.events.CameraSound.Sound"
MOTION_EVENT_KEY = "sdm.devices.events.CameraMotion.Motion"
PERSON_EVENT_KEY = "sdm.devices.events.CameraPerson.Person"

TRACKED_EVENTS = {SOUND_EVENT_KEY, MOTION_EVENT_KEY, PERSON_EVENT_KEY}

_device_name_cache = {}


def get_refresh_token():
    """Load the OAuth refresh token from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GCP_PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8").strip()


def get_oauth_credentials():
    """Build OAuth2 credentials from the stored refresh token."""
    refresh_token = get_refresh_token()
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
    )
    creds.refresh(requests.Request())
    return creds


def get_sdm_headers(creds):
    """Return authorization headers for SDM API calls."""
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def get_camera_name(device_path, creds):
    """Resolve a device path to a human-readable camera name via the Info trait."""
    if device_path in _device_name_cache:
        return _device_name_cache[device_path]

    url = f"{SDM_BASE_URL}/{device_path}"
    resp = requests.get(url, headers=get_sdm_headers(creds))
    if resp.status_code == 200:
        traits = resp.json().get("traits", {})
        custom_name = traits.get("sdm.devices.traits.Info", {}).get("customName", "")
        name = custom_name if custom_name else device_path.split("/")[-1][:8]
        _device_name_cache[device_path] = name
        return name

    short_id = device_path.split("/")[-1][:8]
    _device_name_cache[device_path] = short_id
    return short_id


def get_event_image_url(device_path, event_id, creds):
    """Call GenerateImage on the SDM API to get a snapshot URL for the event.
    Event images expire 30 seconds after the event is published.
    """
    url = f"{SDM_BASE_URL}/{device_path}:executeCommand"
    body = {
        "command": "sdm.devices.commands.CameraEventImage.GenerateImage",
        "params": {"eventId": event_id},
    }
    resp = requests.post(url, headers=get_sdm_headers(creds), json=body)
    if resp.status_code == 200:
        results = resp.json().get("results", {})
        return results.get("url", "")
    logger.warning("GenerateImage failed: %s %s", resp.status_code, resp.text)
    return ""


def classify_event(event_key):
    """Map SDM event keys to readable labels."""
    labels = {
        SOUND_EVENT_KEY: "Sound Detected",
        MOTION_EVENT_KEY: "Motion Detected",
        PERSON_EVENT_KEY: "Person Detected",
    }
    return labels.get(event_key, event_key)


def append_to_sheet(row_data):
    """Append a row to the configured Google Sheet."""
    refresh_token = get_refresh_token()
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    creds.refresh(requests.Request())

    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    sheet.append_row(row_data, value_input_option="USER_ENTERED")
    logger.info("Appended row to sheet: %s", row_data)


def process_sdm_event(envelope):
    """Parse and process an SDM event from a Pub/Sub push message."""
    message = envelope.get("message", {})
    if not message:
        logger.warning("No message in envelope")
        return

    data_b64 = message.get("data", "")
    if not data_b64:
        logger.warning("No data in message")
        return

    payload = json.loads(base64.b64decode(data_b64).decode("utf-8"))
    logger.info("Received SDM event: %s", json.dumps(payload, indent=2))

    resource_update = payload.get("resourceUpdate", {})
    if not resource_update:
        return

    device_path = resource_update.get("name", "")
    events = resource_update.get("events", {})
    timestamp = payload.get("timestamp", "")

    if not events:
        return

    creds = get_oauth_credentials()

    for event_key, event_data in events.items():
        if event_key not in TRACKED_EVENTS:
            continue

        event_id = event_data.get("eventId", "")
        session_id = event_data.get("eventSessionId", "")
        camera_name = get_camera_name(device_path, creds)
        event_label = classify_event(event_key)

        image_url = ""
        if event_id:
            image_url = get_event_image_url(device_path, event_id, creds)

        row = [
            timestamp,
            camera_name,
            event_label,
            event_id,
            session_id,
            image_url,
            "",
        ]
        append_to_sheet(row)
        logger.info("Logged %s from %s at %s", event_label, camera_name, timestamp)


@app.route("/", methods=["POST"])
def handle_pubsub_push():
    """Receive Pub/Sub push messages containing SDM events."""
    envelope = request.get_json(silent=True)
    if not envelope:
        return "Bad Request: no JSON body", 400

    try:
        process_sdm_event(envelope)
    except Exception:
        logger.exception("Error processing event")
        return "Internal Server Error", 500

    return "OK", 200


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run."""
    return "Nest Bark Logger is running", 200


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
