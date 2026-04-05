# DogBark

Automated dog bark detector using a Google Nest camera and the Smart Device Management (SDM) API.
Captures camera sound events in real time, classifies the audio with YAMNet to determine whether the
sound is a dog barking, and logs each event to a Google Sheet -- all running serverless on Google Cloud Run.

Confirmed dog barking events are recorded as permanent WAV audio clips in Cloud Storage.
A Summary tab tracks barking sessions with duration and local time, plus auto-updating daily
and weekly charts.

## Architecture

```
Nest Camera  -->  Google SDM  -->  Cloud Pub/Sub (push)  -->  Cloud Run (Flask)
                                                                  |
                                                        +---------+---------+
                                                        |                   |
                                                  RTSP stream         Cloud Storage
                                                  via FFmpeg          (WAV upload)
                                                        |                   |
                                                  YAMNet TFLite            |
                                                  classification           |
                                                        |                   |
                                                        +----> Google Sheet <+
                                                               |         |
                                                            Sheet1    Summary
                                                          (raw log)  (sessions
                                                                      + charts)
```

When the Nest camera detects a sound, SDM publishes a `CameraSound.Sound` event to a Pub/Sub topic.
Pub/Sub pushes the event (authenticated via OIDC) to the Cloud Run service, which:

1. Generates an RTSP live stream from the camera
2. Captures 8 seconds of audio with FFmpeg (16 kHz mono WAV)
3. Runs YAMNet (TensorFlow Lite, 521 AudioSet classes) to classify the sound
4. Tags the event as **Dog Barking** or **Other Sound** based on dog-related class scores
5. For dog barking events, uploads the WAV clip to Cloud Storage for permanent playback
6. Appends a row to Sheet1 with timestamp, camera name, event type, classification details, and audio URL
7. Updates the Summary tab with session duration tracking

Motion and person detection events are also logged without audio classification.

## Google Sheet

### Sheet1 -- Raw Event Log

Every camera event gets a row here.

| Column | Description |
|--------|-------------|
| Timestamp (UTC) | When the camera detected the event |
| Camera Name | Device name from the Nest Info trait |
| Event Type | `Dog Barking`, `Other Sound`, `Motion Detected`, or `Person Detected` |
| Event ID | SDM event identifier |
| Event Session ID | Groups related events from the same session |
| Audio URL | Cloud Storage link to the 8s WAV clip (dog barking events only) |
| Notes | YAMNet top classification and confidence scores |

### Summary -- Session Tracking and Charts

Bark events are grouped into **sessions**. If two barks are less than 5 minutes apart, they
belong to the same session. If 5+ minutes pass without barking, a new session starts.

| Column | Description |
|--------|-------------|
| Date | Local date (Pacific time) |
| Start Time | When the barking session started, e.g. `3:16 PM` |
| End Time | When the last bark in the session occurred |
| Duration (min) | Total minutes from start to end of the session |
| Bark Count | Number of individual bark events in the session |
| Week | ISO week label, e.g. `2026-W14` |

Two embedded column charts update automatically as new sessions are logged:

- **Daily Bark Minutes** -- total minutes of barking per day
- **Weekly Bark Minutes** -- total minutes of barking per week

The charts are driven by `UNIQUE` + `SUMIF` helper formulas in columns H-L that
aggregate the session data. No manual maintenance required.

## Similar Projects

- [jatacid/dogbarkingdetector](https://github.com/jatacid/dogbarkingdetector) --
  Browser-based dog bark detector using YAMNet via TensorFlow.js. Runs client-side
  with microphone input. No Nest camera integration.
- [MalcolmMielle/bark_monitor](https://codeberg.org/MalcolmMielle/bark_monitor) --
  Python tool that records audio while you're away and detects barking via YAMNet.
  Sends notifications via Matrix/Telegram. Uses a local microphone, not a camera API.
- [tlynam/calm-barking-dog](https://github.com/tlynam/calm-barking-dog) --
  Plays calming sounds to a dog when barking is detected.
- [dogbarkingdetector.com](https://dogbarkingdetector.com/) --
  Online tool using YAMNet for browser-based bark detection.

DogBark differs from these by using the Nest camera's built-in microphone via the SDM API
RTSP stream, running serverless on Cloud Run (no local hardware needed), and maintaining
a structured Google Sheet with session tracking and trend charts.

## Prerequisites

- Google Cloud project with billing enabled
- Device Access registration ($5 one-time fee) at https://console.nest.google.com/device-access
- A legacy Nest camera (Nest Cam Indoor/Outdoor/IQ) with the `CameraSound` and `CameraLiveStream` traits
- Python 3.11+, Docker

## GCP services used

| Service | Purpose |
|---------|---------|
| Smart Device Management API | Camera events and RTSP streams |
| Cloud Pub/Sub | Async event delivery from SDM to Cloud Run |
| Cloud Run | Serverless container hosting the Flask app |
| Cloud Storage | Stores WAV audio clips of dog barking events |
| Secret Manager | Stores the OAuth refresh token |
| Artifact Registry | Stores the Docker container image |
| Google Sheets API | Writes event log rows and session tracking |

## Project structure

```
.
├── nest_dog_bark_logger.py   # Flask app: SDMClient, AudioClassifier, SheetLogger classes
├── yamnet_classify.py        # YAMNet TFLite inference for dog bark detection
├── auth_setup.py             # One-time local script for OAuth consent flow
├── Dockerfile                # Container: Python 3.11, FFmpeg, YAMNet model
├── requirements.txt          # Python dependencies
├── credentials.json          # OAuth client credentials (git-ignored)
├── .env                      # Local environment variables (git-ignored)
└── .gitignore
```

## Setup

### 1. Google Cloud Console

Enable these APIs in your GCP project:

- Smart Device Management API
- Cloud Pub/Sub API
- Google Sheets API
- Cloud Run Admin API
- Secret Manager API
- Artifact Registry API

### 2. Device Access Console

At https://console.nest.google.com/device-access:

- Create a project and link your OAuth client ID
- Enable events with a Pub/Sub topic: `projects/{gcp-project}/topics/{topic-name}`

### 3. Pub/Sub topic

```bash
gcloud pubsub topics create nest-sound-events \
  --project=PROJECT_ID \
  --message-retention-duration=31d

gcloud pubsub topics add-iam-policy-binding \
  projects/PROJECT_ID/topics/nest-sound-events \
  --member="group:sdm-publisher@googlegroups.com" \
  --role="roles/pubsub.publisher"
```

### 4. OAuth consent

Create a `.env` file from the template:

```
GCP_PROJECT_ID=your-gcp-project-id
DEVICE_ACCESS_PROJECT_ID=your-device-access-uuid
OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
OAUTH_CLIENT_SECRET=your-client-secret
SPREADSHEET_ID=your-google-sheet-id
SHEET_NAME=Sheet1
```

Run the one-time setup script to complete the OAuth flow, store the refresh token in
Secret Manager, create sheet headers, and initiate event delivery:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 auth_setup.py
```

The script prints a URL to open in your browser. Grant permissions for home info, camera
events, camera snapshots, and camera livestream. Paste the authorization code back when prompted.

### 5. Deploy to Cloud Run

```bash
gcloud run deploy nest-bark-logger \
  --project=PROJECT_ID \
  --region=us-central1 \
  --source=. \
  --no-allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=...,DEVICE_ACCESS_PROJECT_ID=...,OAUTH_CLIENT_ID=...,OAUTH_CLIENT_SECRET=...,SPREADSHEET_ID=...,SHEET_NAME=Sheet1,AUDIO_BUCKET=dogbark-audio-clips" \
  --memory=512Mi \
  --timeout=120 \
  --min-instances=0 \
  --max-instances=1
```

### 6. Authenticated Pub/Sub push

Create a service account for Pub/Sub to authenticate with Cloud Run:

```bash
gcloud iam service-accounts create pubsub-push-sa \
  --project=PROJECT_ID \
  --display-name="Pub/Sub Push to Cloud Run"

gcloud run services add-iam-policy-binding nest-bark-logger \
  --project=PROJECT_ID \
  --region=us-central1 \
  --member="serviceAccount:pubsub-push-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

PROJECT_NUM=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:service-${PROJECT_NUM}@gcp-sa-pubsub.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator"
```

Grant the Cloud Run default service account access to Secret Manager:

```bash
gcloud secrets add-iam-policy-binding nest-oauth-refresh-token \
  --project=PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Create the push subscription with OIDC authentication:

```bash
CLOUD_RUN_URL=$(gcloud run services describe nest-bark-logger \
  --project=PROJECT_ID --region=us-central1 --format='value(status.url)')

gcloud pubsub subscriptions create nest-sound-push \
  --project=PROJECT_ID \
  --topic=nest-sound-events \
  --push-endpoint="$CLOUD_RUN_URL" \
  --push-auth-service-account="pubsub-push-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --push-auth-token-audience="$CLOUD_RUN_URL" \
  --ack-deadline=60
```

### 7. Initiate event delivery

Call `devices.list` once to start SDM event flow (the `auth_setup.py` script does this automatically).

## Audio classification

YAMNet is a pre-trained MobileNet-v1 model that classifies 521 audio event types from the
AudioSet corpus. The TFLite model (~3.7 MB) runs inference on 16 kHz mono audio.

Dog-related classes monitored (indices 70-75):

| Index | Class |
|-------|-------|
| 70 | Bark |
| 71 | Yip |
| 72 | Howl |
| 73 | Bow-wow |
| 74 | Growling |
| 75 | Whimper |

An event is classified as **Dog Barking** if any dog-related class scores above the confidence
threshold (default 0.25). The threshold can be adjusted in `yamnet_classify.py` via the
`DOG_CONFIDENCE_THRESHOLD` constant.

## Environment variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `DEVICE_ACCESS_PROJECT_ID` | UUID from Device Access Console |
| `OAUTH_CLIENT_ID` | OAuth 2.0 client ID |
| `OAUTH_CLIENT_SECRET` | OAuth 2.0 client secret |
| `SPREADSHEET_ID` | Google Sheet ID (from the URL) |
| `SHEET_NAME` | Worksheet name (default: `Sheet1`) |
| `AUDIO_BUCKET` | Cloud Storage bucket for WAV clips (default: `dogbark-audio-clips`) |

## Cost

The service runs on Cloud Run with `min-instances=0`, so it only incurs cost when processing events.
A single camera generating a few dozen sound events per day stays well within Cloud Run's free tier
(2 million requests/month, 360,000 GB-seconds). Pub/Sub, Secret Manager, Cloud Storage, and Sheets
API usage are similarly minimal. The only fixed cost is the $5 one-time Device Access registration fee.

## Limitations

- The SDM API only supports legacy Nest cameras (Nest Cam Indoor/Outdoor/IQ, Nest Doorbell legacy,
  Nest Hub Max). Newer 2021+ cameras do not expose the `CameraSound` or `CameraLiveStream` traits.
- Audio clips are only saved for events classified as Dog Barking (Other Sound events are not stored).
- Audio classification depends on the RTSP stream being available. If the camera is offline or
  the stream fails, the event is logged as "Sound Detected" with a note that audio capture failed.
- The OAuth refresh token must be used within 6 months or Google may revoke it. Each Cloud Run
  invocation refreshes the access token, which keeps the refresh token alive.
- SDM may suppress rapid repeated similar events (event filtering).
- Session gap threshold is 5 minutes by default. Adjust `SESSION_GAP_MINUTES` in
  `nest_dog_bark_logger.py` if needed.

## License

MIT
