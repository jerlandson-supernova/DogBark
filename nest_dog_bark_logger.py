import base64
import json
import logging
import math
import os
import subprocess
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

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
AUDIO_CAPTURE_SECONDS = 8
AUDIO_SAMPLE_RATE = 16000
FFMPEG_TIMEOUT_SECONDS = 30
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
SUMMARY_SHEET_NAME = "Summary"
SESSION_GAP_MINUTES = 5

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
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", "1",
            wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            logger.warning("FFmpeg failed: %s", proc.stderr.decode(errors="replace")[-500:])
            os.unlink(wav_path)
            return None, stream_token
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timed out after %ds", FFMPEG_TIMEOUT_SECONDS)
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


def get_sheets_client():
    """Get an authenticated gspread client."""
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
    return gspread.authorize(creds)


def setup_summary_sheet(spreadsheet):
    """Create the Summary worksheet with headers and helper formulas if it doesn't exist."""
    try:
        spreadsheet.worksheet(SUMMARY_SHEET_NAME)
        return
    except gspread.exceptions.WorksheetNotFound:
        pass

    ws = spreadsheet.add_worksheet(title=SUMMARY_SHEET_NAME, rows=1000, cols=12)

    headers = [
        ["Date", "Start Time", "End Time", "Duration (min)", "Bark Count", "Week",
         "", "Date", "Minutes", "", "Week", "Minutes"],
    ]
    ws.update("A1:L1", headers, value_input_option="USER_ENTERED")

    ws.format("A1:L1", {"textFormat": {"bold": True}})

    formulas = [
        ['=IFERROR(SORT(UNIQUE(FILTER(A2:A, A2:A<>""))),"")'],
        ['=ARRAYFORMULA(IF(H2:H="","",SUMIF(A$2:A,H2:H,D$2:D)))'],
        [""],
        ['=IFERROR(SORT(UNIQUE(FILTER(F2:F, F2:F<>""))),"")'],
        ['=ARRAYFORMULA(IF(K2:K="","",SUMIF(F$2:F,K2:K,D$2:D)))'],
    ]
    ws.update("H2:L2", [formulas[0] + [""] + formulas[3]], value_input_option="USER_ENTERED")
    ws.update("I2", [[formulas[1][0]]], value_input_option="USER_ENTERED")
    ws.update("L2", [[formulas[4][0]]], value_input_option="USER_ENTERED")

    create_summary_charts(spreadsheet, ws)
    logger.info("Created Summary sheet with headers, formulas, and charts")


def create_summary_charts(spreadsheet, summary_ws):
    """Add daily and weekly bark-minutes charts to the Summary sheet."""
    sheet_id = summary_ws.id

    daily_chart = {
        "addChart": {
            "chart": {
                "position": {"overlayPosition": {"anchorCell": {"sheetId": sheet_id, "rowIndex": 3, "columnIndex": 7}}},
                "spec": {
                    "title": "Daily Bark Minutes",
                    "basicChart": {
                        "chartType": "COLUMN",
                        "legendPosition": "NO_LEGEND",
                        "axis": [
                            {"position": "BOTTOM_AXIS", "title": "Date"},
                            {"position": "LEFT_AXIS", "title": "Minutes"},
                        ],
                        "domains": [{"domain": {"sourceRange": {"sources": [
                            {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 500,
                             "startColumnIndex": 7, "endColumnIndex": 8}
                        ]}}}],
                        "series": [{"series": {"sourceRange": {"sources": [
                            {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 500,
                             "startColumnIndex": 8, "endColumnIndex": 9}
                        ]}}, "targetAxis": "LEFT_AXIS"}],
                        "headerCount": 1,
                    },
                },
            }
        }
    }

    weekly_chart = {
        "addChart": {
            "chart": {
                "position": {"overlayPosition": {"anchorCell": {"sheetId": sheet_id, "rowIndex": 20, "columnIndex": 7}}},
                "spec": {
                    "title": "Weekly Bark Minutes",
                    "basicChart": {
                        "chartType": "COLUMN",
                        "legendPosition": "NO_LEGEND",
                        "axis": [
                            {"position": "BOTTOM_AXIS", "title": "Week"},
                            {"position": "LEFT_AXIS", "title": "Minutes"},
                        ],
                        "domains": [{"domain": {"sourceRange": {"sources": [
                            {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 500,
                             "startColumnIndex": 10, "endColumnIndex": 11}
                        ]}}}],
                        "series": [{"series": {"sourceRange": {"sources": [
                            {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 500,
                             "startColumnIndex": 11, "endColumnIndex": 12}
                        ]}}, "targetAxis": "LEFT_AXIS"}],
                        "headerCount": 1,
                    },
                },
            }
        }
    }

    spreadsheet.batch_update({"requests": [daily_chart, weekly_chart]})


def utc_to_local(utc_str):
    """Parse a UTC ISO timestamp and convert to local (Pacific) datetime."""
    utc_str = utc_str.replace("Z", "+00:00")
    dt_utc = datetime.fromisoformat(utc_str)
    return dt_utc.astimezone(LOCAL_TZ)


def update_summary_session(gc, timestamp_utc):
    """Track barking sessions on the Summary sheet.
    Groups bark events into sessions separated by gaps of 5+ minutes.
    """
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    setup_summary_sheet(spreadsheet)
    ws = spreadsheet.worksheet(SUMMARY_SHEET_NAME)

    local_dt = utc_to_local(timestamp_utc)
    local_date = local_dt.strftime("%Y-%m-%d")
    local_time = local_dt.strftime("%-I:%M %p")
    iso_week = f"{local_dt.isocalendar()[0]}-W{local_dt.isocalendar()[1]:02d}"

    all_values = ws.get_all_values()
    data_rows = [r for r in all_values[1:] if r[0]] if len(all_values) > 1 else []

    extend_session = False
    if data_rows:
        last_row = data_rows[-1]
        last_row_idx = len(all_values)
        try:
            last_date = last_row[0]
            last_end_str = last_row[2]
            last_end_dt = datetime.strptime(f"{last_date} {last_end_str}", "%Y-%m-%d %I:%M %p")
            last_end_dt = last_end_dt.replace(tzinfo=LOCAL_TZ)
            gap = (local_dt - last_end_dt).total_seconds() / 60.0
            if gap < SESSION_GAP_MINUTES:
                extend_session = True
        except (ValueError, IndexError):
            pass

    if extend_session:
        start_str = last_row[1]
        start_dt = datetime.strptime(f"{last_row[0]} {start_str}", "%Y-%m-%d %I:%M %p")
        duration = max(1, math.ceil((local_dt - start_dt.replace(tzinfo=LOCAL_TZ)).total_seconds() / 60.0))
        bark_count = int(last_row[4]) + 1 if last_row[4].isdigit() else 2

        row_range = f"A{last_row_idx}:F{last_row_idx}"
        ws.update(row_range, [[last_row[0], start_str, local_time, duration, bark_count, last_row[5]]],
                  value_input_option="USER_ENTERED")
        logger.info("Extended session: %s %s-%s (%d min, %d barks)", last_row[0], start_str, local_time, duration, bark_count)
    else:
        new_row = [local_date, local_time, local_time, 1, 1, iso_week]
        ws.append_row(new_row, value_input_option="USER_ENTERED")
        logger.info("New session: %s %s", local_date, local_time)


def append_to_sheet(row_data):
    """Append a row to the raw events sheet."""
    gc = get_sheets_client()
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    sheet.append_row(row_data, value_input_option="USER_ENTERED")
    logger.info("Appended row to sheet: %s", row_data)
    return gc


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
        gc = append_to_sheet(row)
        logger.info("Logged %s from %s at %s", event_label, camera_name, timestamp)

        if event_label == "Dog Barking":
            try:
                update_summary_session(gc, timestamp)
            except Exception:
                logger.exception("Failed to update summary session")


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
