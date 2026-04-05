import base64
import json
import logging
import os
import subprocess
import tempfile

import gspread
import requests as req_lib
from flask import Flask, request
from google.auth.transport.requests import Request as AuthRequest
from google.cloud import secretmanager, storage
from google.oauth2.credentials import Credentials

from yamnet_classify import is_dog_barking

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nest_bark_logger")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DEVICE_ACCESS_PROJECT_ID = os.environ.get("DEVICE_ACCESS_PROJECT_ID")
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
SHEET_NAME = os.environ.get("SHEET_NAME", "Sheet1")
AUDIO_BUCKET = os.environ.get("AUDIO_BUCKET", "dogbark-audio-clips")

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
    creds.refresh(AuthRequest())
    return creds


def get_sdm_headers(creds):
    """Return authorization headers for SDM API calls."""
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def get_camera_name(device_path, creds):
    """Resolve a device path to a human-readable camera name via the Info trait."""
    if device_path in _device_name_cache:
        return _device_name_cache[device_path]

    url = f"{SDM_BASE_URL}/{device_path}"
    resp = req_lib.get(url, headers=get_sdm_headers(creds))
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
    resp = req_lib.post(url, headers=get_sdm_headers(creds), json=body)
    if resp.status_code == 200:
        results = resp.json().get("results", {})
        return results.get("url", "")
    logger.warning("GenerateImage failed: %s %s", resp.status_code, resp.text)
    return ""


AUDIO_CAPTURE_SECONDS = 8


def capture_audio_from_stream(device_path, creds):
    """Start an RTSP stream, capture audio with FFmpeg, return WAV path and stream token."""
    url = f"{SDM_BASE_URL}/{device_path}:executeCommand"
    body = {
        "command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream",
        "params": {},
    }
    resp = req_lib.post(url, headers=get_sdm_headers(creds), json=body)
    if resp.status_code != 200:
        logger.warning("GenerateRtspStream failed: %s %s", resp.status_code, resp.text)
        return None, None

    results = resp.json().get("results", {})
    rtsp_url = results.get("streamUrls", {}).get("rtspUrl", "")
    stream_token = results.get("streamToken", "")
    if not rtsp_url:
        logger.warning("No RTSP URL in response")
        return None, None

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-t", str(AUDIO_CAPTURE_SECONDS),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0:
            logger.warning("FFmpeg failed: %s", proc.stderr.decode(errors="replace")[-500:])
            os.unlink(wav_path)
            return None, stream_token
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timed out after 30s")
        os.unlink(wav_path)
        return None, stream_token

    return wav_path, stream_token


def stop_rtsp_stream(device_path, stream_token, creds):
    """Stop an active RTSP stream."""
    if not stream_token:
        return
    url = f"{SDM_BASE_URL}/{device_path}:executeCommand"
    body = {
        "command": "sdm.devices.commands.CameraLiveStream.StopRtspStream",
        "params": {"streamExtensionToken": stream_token},
    }
    req_lib.post(url, headers=get_sdm_headers(creds), json=body)


def upload_audio_to_gcs(wav_path, timestamp):
    """Upload a WAV file to Cloud Storage. Returns the public URL."""
    safe_ts = timestamp.replace(":", "-").replace(".", "-")
    blob_name = f"barks/{safe_ts}.wav"
    client = storage.Client()
    bucket = client.bucket(AUDIO_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(wav_path, content_type="audio/wav")
    return f"https://storage.googleapis.com/{AUDIO_BUCKET}/{blob_name}"


def classify_sound_event(device_path, creds, timestamp):
    """Capture audio from camera and classify with YAMNet.
    Returns (event_label, notes_text, audio_url).
    """
    wav_path, stream_token = capture_audio_from_stream(device_path, creds)
    try:
        if wav_path is None:
            return "Sound Detected", "Audio capture failed", ""

        is_dog, top_class, top_conf, dog_conf = is_dog_barking(wav_path)
        audio_url = ""
        if is_dog:
            label = "Dog Barking"
            notes = f"YAMNet: {top_class} ({top_conf:.0%}), dog_score={dog_conf:.0%}"
            try:
                audio_url = upload_audio_to_gcs(wav_path, timestamp)
                logger.info("Uploaded bark audio: %s", audio_url)
            except Exception:
                logger.exception("Failed to upload audio to GCS")
        else:
            label = "Other Sound"
            notes = f"YAMNet: {top_class} ({top_conf:.0%}), dog_score={dog_conf:.0%}"
        return label, notes, audio_url
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        stop_rtsp_stream(device_path, stream_token, creds)


def classify_event(event_key):
    """Map SDM event keys to readable labels for non-sound events."""
    labels = {
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
    creds.refresh(AuthRequest())

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

        audio_url = ""
        notes = ""
        if event_key == SOUND_EVENT_KEY:
            event_label, notes, audio_url = classify_sound_event(device_path, creds, timestamp)
        else:
            event_label = classify_event(event_key)

        row = [
            timestamp,
            camera_name,
            event_label,
            event_id,
            session_id,
            audio_url,
            notes,
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
