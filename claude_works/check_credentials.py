"""
Quick check: tests each credential file for TTS API access.
"""
import json
import os
from pathlib import Path
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.oauth2 import service_account

CREDENTIALS_FOLDER = Path("claude_works/credentials")
API_ENDPOINT = "texttospeech.googleapis.com"
MODEL = "gemini-2.5-pro-tts"

AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.0,
    pitch=0
)

VOICE = texttospeech.VoiceSelectionParams(
    name="Zephyr",
    language_code="kn-IN",
    model_name=MODEL
)

def test_credential(path, label):
    try:
        creds = service_account.Credentials.from_service_account_file(
            str(path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = texttospeech.TextToSpeechClient(
            credentials=creds,
            client_options=ClientOptions(api_endpoint=API_ENDPOINT)
        )
        resp = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text="test"),
            voice=VOICE,
            audio_config=AUDIO_CONFIG
        )
        audio_bytes = len(resp.audio_content)
        print(f"  OK    {label} — got {audio_bytes} bytes")
        return True
    except Exception as e:
        err = str(e).split("\n")[0][:120]
        print(f"  FAIL  {label} — {err}")
        return False

def main():
    cred_files = []

    env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_cred and Path(env_cred).exists():
        cred_files.append((Path(env_cred), "ENV"))

    for f in sorted(CREDENTIALS_FOLDER.glob("*.json")):
        cred_files.append((f, f.name))

    if not cred_files:
        print("No credential files found!")
        return

    print(f"Testing {len(cred_files)} credentials...\n")

    ok = 0
    fail = 0
    for path, label in cred_files:
        with open(path) as f:
            project_id = json.load(f).get("project_id", "?")
        full_label = f"{label:50s} (project: {project_id})"
        if test_credential(path, full_label):
            ok += 1
        else:
            fail += 1

    print(f"\nResults: {ok} OK, {fail} FAILED out of {len(cred_files)}")

if __name__ == "__main__":
    main()
