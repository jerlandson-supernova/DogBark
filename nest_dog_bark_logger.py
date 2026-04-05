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

# ---------------------------------------------------------------------------
# GCP project and OAuth credentials (from environment / Cloud Run config)
# ---------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DEVICE_ACCESS_PROJECT_ID = os.environ.get("DEVICE_ACCESS_PROJECT_ID")
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")

# ---------------------------------------------------------------------------
# Google Sheets configuration
# ---------------------------------------------------------------------------
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID")
SHEET_NAME = os.environ.get("SHEET_NAME", "Sheet1")
SUMMARY_SHEET_NAME = "Summary"

# ---------------------------------------------------------------------------
# Audio capture and classification
# ---------------------------------------------------------------------------
# Cloud Storage bucket for permanent WAV clips of confirmed dog barking
AUDIO_BUCKET = os.environ.get("AUDIO_BUCKET", "dogbark-audio-clips")
# Duration of RTSP audio capture per sound event
AUDIO_CAPTURE_SECONDS = 8
# Sample rate for FFmpeg output and YAMNet input
AUDIO_SAMPLE_RATE = 16000
# Max time to wait for FFmpeg before killing the process
FFMPEG_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Session tracking (Summary tab)
# ---------------------------------------------------------------------------
# Timezone for local time display on the Summary sheet
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
# If two bark events are less than this many minutes apart, they belong
# to the same barking session; otherwise a new session row is created
SESSION_GAP_MINUTES = 5

# ---------------------------------------------------------------------------
# SDM API
# ---------------------------------------------------------------------------
SDM_BASE_URL = "https://smartdevicemanagement.googleapis.com/v1"
TOKEN_URI = "https://oauth2.googleapis.com/token"
SECRET_NAME = "nest-oauth-refresh-token"

# SDM event type keys
SOUND_EVENT_KEY = "sdm.devices.events.CameraSound.Sound"
MOTION_EVENT_KEY = "sdm.devices.events.CameraMotion.Motion"
PERSON_EVENT_KEY = "sdm.devices.events.CameraPerson.Person"

# Only these event types are logged to the sheet
TRACKED_EVENTS = {SOUND_EVENT_KEY, MOTION_EVENT_KEY, PERSON_EVENT_KEY}

# Readable labels for non-sound events
EVENT_LABELS = {
    MOTION_EVENT_KEY: "Motion Detected",
    PERSON_EVENT_KEY: "Person Detected",
}

# ---------------------------------------------------------------------------
# Flask app and logging
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nest_bark_logger")


# ===========================================================================
# SDMClient -- handles all Smart Device Management API interactions
# ===========================================================================
class SDMClient:
    """Manages OAuth credentials and SDM API calls for Nest camera devices."""

    def __init__(self):
        self._device_name_cache = {}

    def _get_refresh_token(self):
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()

    def get_credentials(self):
        """Build and refresh OAuth2 credentials from the stored refresh token."""
        refresh_token = self._get_refresh_token()
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET,
        )
        creds.refresh(AuthRequest())
        return creds

    def get_sheets_credentials(self):
        """Build OAuth2 credentials scoped for the Google Sheets API."""
        refresh_token = self._get_refresh_token()
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        creds.refresh(AuthRequest())
        return creds

    def _headers(self, creds):
        return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}

    def _execute_command(self, device_path, command, params, creds):
        url = f"{SDM_BASE_URL}/{device_path}:executeCommand"
        body = {"command": command, "params": params}
        return req_lib.post(url, headers=self._headers(creds), json=body)

    def get_camera_name(self, device_path, creds):
        """Resolve a device path to a human-readable name via the Info trait."""
        if device_path in self._device_name_cache:
            return self._device_name_cache[device_path]

        url = f"{SDM_BASE_URL}/{device_path}"
        resp = req_lib.get(url, headers=self._headers(creds))
        if resp.status_code == 200:
            traits = resp.json().get("traits", {})
            custom_name = traits.get("sdm.devices.traits.Info", {}).get("customName", "")
            name = custom_name if custom_name else device_path.split("/")[-1][:8]
        else:
            name = device_path.split("/")[-1][:8]

        self._device_name_cache[device_path] = name
        return name

    def generate_rtsp_stream(self, device_path, creds):
        """Request an RTSP live stream URL from the camera.
        Returns (rtsp_url, stream_token) or (None, None) on failure.
        """
        resp = self._execute_command(
            device_path,
            "sdm.devices.commands.CameraLiveStream.GenerateRtspStream",
            {},
            creds,
        )
        if resp.status_code != 200:
            logger.warning("GenerateRtspStream failed: %s %s", resp.status_code, resp.text)
            return None, None

        results = resp.json().get("results", {})
        rtsp_url = results.get("streamUrls", {}).get("rtspUrl", "")
        stream_token = results.get("streamToken", "")
        if not rtsp_url:
            logger.warning("No RTSP URL in response")
            return None, None
        return rtsp_url, stream_token

    def stop_rtsp_stream(self, device_path, stream_token, creds):
        """Stop an active RTSP stream."""
        if not stream_token:
            return
        self._execute_command(
            device_path,
            "sdm.devices.commands.CameraLiveStream.StopRtspStream",
            {"streamExtensionToken": stream_token},
            creds,
        )


# ===========================================================================
# AudioClassifier -- captures camera audio and classifies with YAMNet
# ===========================================================================
class AudioClassifier:
    """Captures audio from a Nest camera RTSP stream and classifies it."""

    def __init__(self, sdm_client):
        self._sdm = sdm_client

    def _capture_wav(self, rtsp_url):
        """Use FFmpeg to record audio from an RTSP stream as a 16 kHz mono WAV.
        Returns the path to the WAV file, or None on failure.
        """
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

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
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
            if proc.returncode != 0:
                logger.warning("FFmpeg failed: %s", proc.stderr.decode(errors="replace")[-500:])
                os.unlink(wav_path)
                return None
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg timed out after %ds", FFMPEG_TIMEOUT_SECONDS)
            os.unlink(wav_path)
            return None

        return wav_path

    @staticmethod
    def _upload_to_gcs(wav_path, timestamp):
        """Upload a WAV file to Cloud Storage and return the permanent URL."""
        safe_ts = timestamp.replace(":", "-").replace(".", "-")
        blob_name = f"barks/{safe_ts}.wav"
        client = storage.Client()
        bucket = client.bucket(AUDIO_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(wav_path, content_type="audio/wav")
        return f"https://storage.googleapis.com/{AUDIO_BUCKET}/{blob_name}"

    def classify(self, device_path, creds, timestamp):
        """Capture audio from the camera and classify the sound.
        Returns (event_label, notes, audio_url).
        """
        rtsp_url, stream_token = self._sdm.generate_rtsp_stream(device_path, creds)
        if rtsp_url is None:
            return "Sound Detected", "Audio capture failed", ""

        wav_path = self._capture_wav(rtsp_url)
        try:
            if wav_path is None:
                return "Sound Detected", "FFmpeg capture failed", ""

            is_dog, top_class, top_conf, dog_conf = is_dog_barking(wav_path)
            notes = f"YAMNet: {top_class} ({top_conf:.0%}), dog_score={dog_conf:.0%}"

            audio_url = ""
            if is_dog:
                label = "Dog Barking"
                try:
                    audio_url = self._upload_to_gcs(wav_path, timestamp)
                    logger.info("Uploaded bark audio: %s", audio_url)
                except Exception:
                    logger.exception("Failed to upload audio to GCS")
            else:
                label = "Other Sound"

            return label, notes, audio_url
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self._sdm.stop_rtsp_stream(device_path, stream_token, creds)


# ===========================================================================
# SheetLogger -- writes events and sessions to Google Sheets
# ===========================================================================
class SheetLogger:
    """Manages all Google Sheets operations: raw event log and summary sessions."""

    def __init__(self, sdm_client):
        self._sdm = sdm_client

    def _get_client(self):
        creds = self._sdm.get_sheets_credentials()
        return gspread.authorize(creds)

    def append_event(self, row_data):
        """Append a row to the raw events sheet. Returns the gspread client for reuse."""
        gc = self._get_client()
        sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        sheet.append_row(row_data, value_input_option="USER_ENTERED")
        logger.info("Appended row to sheet: %s", row_data)
        return gc

    # -- Summary tab -----------------------------------------------------------

    def _ensure_summary_sheet(self, spreadsheet):
        """Create the Summary worksheet with headers, formulas, and charts if missing."""
        try:
            spreadsheet.worksheet(SUMMARY_SHEET_NAME)
            return
        except gspread.exceptions.WorksheetNotFound:
            pass

        ws = spreadsheet.add_worksheet(title=SUMMARY_SHEET_NAME, rows=1000, cols=12)

        headers = [[
            "Date", "Start Time", "End Time", "Duration (min)", "Bark Count", "Week",
            "", "Date", "Minutes", "", "Week", "Minutes",
        ]]
        ws.update("A1:L1", headers, value_input_option="USER_ENTERED")
        ws.format("A1:L1", {"textFormat": {"bold": True}})

        # Daily aggregation formulas (H:I)
        ws.update("H2", [['=IFERROR(SORT(UNIQUE(FILTER(A2:A, A2:A<>""))),"")']],
                  value_input_option="USER_ENTERED")
        ws.update("I2", [['=ARRAYFORMULA(IF(H2:H="","",SUMIF(A$2:A,H2:H,D$2:D)))']],
                  value_input_option="USER_ENTERED")

        # Weekly aggregation formulas (K:L)
        ws.update("K2", [['=IFERROR(SORT(UNIQUE(FILTER(F2:F, F2:F<>""))),"")']],
                  value_input_option="USER_ENTERED")
        ws.update("L2", [['=ARRAYFORMULA(IF(K2:K="","",SUMIF(F$2:F,K2:K,D$2:D)))']],
                  value_input_option="USER_ENTERED")

        self._create_charts(spreadsheet, ws)
        logger.info("Created Summary sheet with headers, formulas, and charts")

    @staticmethod
    def _create_charts(spreadsheet, summary_ws):
        """Embed daily and weekly bark-minutes column charts on the Summary sheet."""
        sid = summary_ws.id

        def _chart_request(title, anchor_row, domain_col, series_col):
            return {
                "addChart": {
                    "chart": {
                        "position": {"overlayPosition": {
                            "anchorCell": {"sheetId": sid, "rowIndex": anchor_row, "columnIndex": 7},
                        }},
                        "spec": {
                            "title": title,
                            "basicChart": {
                                "chartType": "COLUMN",
                                "legendPosition": "NO_LEGEND",
                                "axis": [
                                    {"position": "BOTTOM_AXIS", "title": title.split()[0]},
                                    {"position": "LEFT_AXIS", "title": "Minutes"},
                                ],
                                "domains": [{"domain": {"sourceRange": {"sources": [{
                                    "sheetId": sid, "startRowIndex": 0, "endRowIndex": 500,
                                    "startColumnIndex": domain_col, "endColumnIndex": domain_col + 1,
                                }]}}}],
                                "series": [{"series": {"sourceRange": {"sources": [{
                                    "sheetId": sid, "startRowIndex": 0, "endRowIndex": 500,
                                    "startColumnIndex": series_col, "endColumnIndex": series_col + 1,
                                }]}}, "targetAxis": "LEFT_AXIS"}],
                                "headerCount": 1,
                            },
                        },
                    }
                }
            }

        spreadsheet.batch_update({"requests": [
            _chart_request("Daily Bark Minutes", anchor_row=3, domain_col=7, series_col=8),
            _chart_request("Weekly Bark Minutes", anchor_row=20, domain_col=10, series_col=11),
        ]})

    @staticmethod
    def _utc_to_local(utc_str):
        utc_str = utc_str.replace("Z", "+00:00")
        return datetime.fromisoformat(utc_str).astimezone(LOCAL_TZ)

    def update_session(self, gc, timestamp_utc):
        """Track barking sessions on the Summary sheet.
        Groups bark events into sessions separated by gaps of SESSION_GAP_MINUTES.
        """
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
        self._ensure_summary_sheet(spreadsheet)
        ws = spreadsheet.worksheet(SUMMARY_SHEET_NAME)

        local_dt = self._utc_to_local(timestamp_utc)
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
                last_end_dt = datetime.strptime(
                    f"{last_row[0]} {last_row[2]}", "%Y-%m-%d %I:%M %p"
                ).replace(tzinfo=LOCAL_TZ)
                gap = (local_dt - last_end_dt).total_seconds() / 60.0
                if gap < SESSION_GAP_MINUTES:
                    extend_session = True
            except (ValueError, IndexError):
                pass

        if extend_session:
            start_str = last_row[1]
            start_dt = datetime.strptime(
                f"{last_row[0]} {start_str}", "%Y-%m-%d %I:%M %p"
            ).replace(tzinfo=LOCAL_TZ)
            duration = max(1, math.ceil((local_dt - start_dt).total_seconds() / 60.0))
            bark_count = int(last_row[4]) + 1 if last_row[4].isdigit() else 2

            ws.update(
                f"A{last_row_idx}:F{last_row_idx}",
                [[last_row[0], start_str, local_time, duration, bark_count, last_row[5]]],
                value_input_option="USER_ENTERED",
            )
            logger.info(
                "Extended session: %s %s-%s (%d min, %d barks)",
                last_row[0], start_str, local_time, duration, bark_count,
            )
        else:
            ws.append_row(
                [local_date, local_time, local_time, 1, 1, iso_week],
                value_input_option="USER_ENTERED",
            )
            logger.info("New session: %s %s", local_date, local_time)


# ===========================================================================
# Module-level instances (created once per container cold start)
# ===========================================================================
sdm_client = SDMClient()
audio_classifier = AudioClassifier(sdm_client)
sheet_logger = SheetLogger(sdm_client)


# ===========================================================================
# Event processing pipeline
# ===========================================================================
def process_sdm_event(envelope):
    """Parse a Pub/Sub push message and route the SDM event to the appropriate handler."""
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

    creds = sdm_client.get_credentials()

    for event_key, event_data in events.items():
        if event_key not in TRACKED_EVENTS:
            continue

        event_id = event_data.get("eventId", "")
        session_id = event_data.get("eventSessionId", "")
        camera_name = sdm_client.get_camera_name(device_path, creds)

        audio_url = ""
        notes = ""
        if event_key == SOUND_EVENT_KEY:
            event_label, notes, audio_url = audio_classifier.classify(device_path, creds, timestamp)
        else:
            event_label = EVENT_LABELS.get(event_key, event_key)

        row = [timestamp, camera_name, event_label, event_id, session_id, audio_url, notes]
        gc = sheet_logger.append_event(row)
        logger.info("Logged %s from %s at %s", event_label, camera_name, timestamp)

        if event_label == "Dog Barking":
            try:
                sheet_logger.update_session(gc, timestamp)
            except Exception:
                logger.exception("Failed to update summary session")


# ===========================================================================
# Flask routes
# ===========================================================================
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
