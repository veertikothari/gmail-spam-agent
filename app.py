from flask import Flask, request, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os, joblib, base64, re
from email import message_from_bytes
from apscheduler.schedulers.background import BackgroundScheduler

# SCOPES for Gmail modify
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Load ML Model
MODEL_PATH = "spam_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)   # load TF-IDF vectorizer
model = joblib.load(MODEL_PATH)             # load trained classifier

stats = {"spam": 0, "ham": 0, "total": 0}

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Gmail Spam Agent API is running!"})

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def classify_message(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
    raw_data = base64.urlsafe_b64decode(msg['raw'])
    email_msg = message_from_bytes(raw_data)
    subject = email_msg.get('Subject', '')
    sender = email_msg.get('From', '')
    body = ""
    if email_msg.is_multipart():
        for part in email_msg.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload(decode=True).decode(errors="ignore")
    else:
        body = email_msg.get_payload(decode=True).decode(errors="ignore")

    text = subject + " " + body

    # RULE: auto spam if from Zoom
    if "no-reply@zoom.us" in sender.lower():
        label_spam(service, msg_id)
        return "spam (zoom rule)"

    # ML prediction
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    if prediction == 1:  # spam
        label_spam(service, msg_id)
        return "spam"
    else:
        return "ham"

def label_spam(service, msg_id):
    # Ensure Spam-ML label exists
    label_name = "Spam-ML"
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    spam_label_id = None
    for l in labels:
        if l['name'] == label_name:
            spam_label_id = l['id']
            break
    if not spam_label_id:
        new_label = {
            "name": label_name,
            "messageListVisibility": "show",
            "labelListVisibility": "labelShow"
        }
        created = service.users().labels().create(userId='me', body=new_label).execute()
        spam_label_id = created['id']

    service.users().messages().modify(
        userId='me',
        id=msg_id,
        body={"addLabelIds": [spam_label_id]}
    ).execute()

def check_emails():
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    output = []
    for m in messages:
        result = classify_message(service, m['id'])
        output.append({"id": m['id'], "result": result})
    return output

@app.route("/check-emails", methods=["GET"])
def api_check_emails():
    results = check_emails()
    return jsonify(results)

@app.route("/spam-stats", methods=["GET"])
def spam_stats():
    service = get_gmail_service()
    label_id = None
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    for l in labels:
        if l['name'] == "Spam-ML":
            label_id = l['id']
            break

    if not label_id:
        return jsonify({"spam": 0, "ham": "unknown"})

    results = service.users().messages().list(userId='me', labelIds=[label_id]).execute()
    spam_count = len(results.get('messages', []))
    return jsonify({"spam": spam_count})

# Background job (every 5 mins)
scheduler = BackgroundScheduler()
scheduler.add_job(func=check_emails, trigger="interval", minutes=5)
scheduler.start()


if __name__ == "__main__":
    app.run(port=5000, debug=True)

