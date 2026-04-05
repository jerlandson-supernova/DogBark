"""One-time local script to complete the Nest SDM OAuth flow.

Run this once on your local machine to:
1. Generate the authorization URL (PCM)
2. Exchange the auth code for tokens
3. Store the refresh token in Google Secret Manager
4. Call devices.list to initiate SDM event delivery
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from google.cloud import secretmanager

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DEVICE_ACCESS_PROJECT_ID = os.environ.get("DEVICE_ACCESS_PROJECT_ID")
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")

REDIRECT_URI = "https://www.google.com"
TOKEN_URI = "https://oauth2.googleapis.com/token"
SDM_SCOPE = "https://www.googleapis.com/auth/sdm.service"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
SECRET_NAME = "nest-oauth-refresh-token"

SDM_BASE_URL = "https://smartdevicemanagement.googleapis.com/v1"


def build_auth_url():
    """Build the Partner Connections Manager (PCM) authorization URL."""
    params = {
        "redirect_uri": REDIRECT_URI,
        "client_id": OAUTH_CLIENT_ID,
        "access_type": "offline",
        "prompt": "consent",
        "response_type": "code",
        "scope": f"{SDM_SCOPE} {SHEETS_SCOPE}",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://nestservices.google.com/partnerconnections/{DEVICE_ACCESS_PROJECT_ID}/auth?{query}"


def exchange_code_for_tokens(auth_code):
    """Exchange the authorization code for access and refresh tokens."""
    data = {
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    resp = requests.post(TOKEN_URI, data=data)
    if resp.status_code != 200:
        print(f"Token exchange failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()


def store_refresh_token(refresh_token):
    """Store the refresh token in Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{GCP_PROJECT_ID}"
    secret_path = f"{parent}/secrets/{SECRET_NAME}"

    try:
        client.get_secret(request={"name": secret_path})
        print(f"Secret '{SECRET_NAME}' already exists, adding new version...")
    except Exception:
        print(f"Creating secret '{SECRET_NAME}'...")
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": SECRET_NAME,
                "secret": {"replication": {"automatic": {}}},
            }
        )

    client.add_secret_version(
        request={
            "parent": secret_path,
            "payload": {"data": refresh_token.encode("UTF-8")},
        }
    )
    print(f"Refresh token stored in Secret Manager as '{SECRET_NAME}'")


def list_devices(access_token):
    """Call devices.list to initiate SDM event delivery and show available devices."""
    url = f"{SDM_BASE_URL}/enterprises/{DEVICE_ACCESS_PROJECT_ID}/devices"
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"devices.list failed: {resp.status_code} {resp.text}")
        return

    devices = resp.json().get("devices", [])
    print(f"\nFound {len(devices)} device(s):")
    for device in devices:
        name = device.get("name", "")
        device_type = device.get("type", "")
        traits = device.get("traits", {})
        custom_name = traits.get("sdm.devices.traits.Info", {}).get("customName", "unknown")
        has_sound = "sdm.devices.traits.CameraSound" in traits

        print(f"  - {custom_name} ({device_type})")
        print(f"    Path: {name}")
        print(f"    Sound detection: {'YES' if has_sound else 'NO'}")

    print("\nEvent delivery has been initiated. Events will now flow to your Pub/Sub topic.")


def main():
    missing = []
    if not GCP_PROJECT_ID:
        missing.append("GCP_PROJECT_ID")
    if not DEVICE_ACCESS_PROJECT_ID:
        missing.append("DEVICE_ACCESS_PROJECT_ID")
    if not OAUTH_CLIENT_ID:
        missing.append("OAUTH_CLIENT_ID")
    if not OAUTH_CLIENT_SECRET:
        missing.append("OAUTH_CLIENT_SECRET")

    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Set them in your .env file and try again.")
        sys.exit(1)

    if DEVICE_ACCESS_PROJECT_ID == "REPLACE_WITH_DEVICE_ACCESS_PROJECT_UUID":
        print("You need to replace DEVICE_ACCESS_PROJECT_ID in your .env file")
        print("with the UUID from the Device Access Console (Phase 3).")
        sys.exit(1)

    auth_url = build_auth_url()
    print("\n--- Step 1: Authorize ---")
    print("Open this URL in your browser:\n")
    print(auth_url)
    print("\nAfter granting access, you will be redirected to google.com.")
    print("Copy the 'code' parameter from the URL bar.")
    print("It looks like: 4/0AXx...long-string...\n")

    auth_code = input("Paste the authorization code here: ").strip()
    if not auth_code:
        print("No code provided. Exiting.")
        sys.exit(1)

    print("\n--- Step 2: Exchange code for tokens ---")
    tokens = exchange_code_for_tokens(auth_code)
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    if not refresh_token:
        print("No refresh token received. Make sure you used access_type=offline and prompt=consent.")
        sys.exit(1)

    print(f"Access token received (expires in {tokens.get('expires_in', '?')}s)")
    print("Refresh token received")

    print("\n--- Step 3: Store refresh token in Secret Manager ---")
    store_refresh_token(refresh_token)

    print("\n--- Step 4: List devices and initiate events ---")
    list_devices(access_token)

    print("\nSetup complete. Your Cloud Run service will now receive events.")


if __name__ == "__main__":
    main()
