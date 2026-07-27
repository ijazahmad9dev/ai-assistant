import imaplib
import email
import re
from email.header import decode_header
from app.src.config import IMAP_SERVER, IMAP_PORT, SMTP_EMAIL, SMTP_APP_PASSWORD


def _decode(value):
    if value is None:
        return ""
    parts = decode_header(value)
    decoded = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded += part.decode(enc or "utf-8", errors="ignore")
        else:
            decoded += part
    return decoded


def _get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                return part.get_payload(decode=True).decode(errors="ignore")
        return ""
    return msg.get_payload(decode=True).decode(errors="ignore")


def _guess_request_type(subject: str) -> str:
    subject_lower = subject.lower()
    if "leave request" in subject_lower:
        return "Leave Request"
    if "wfh request" in subject_lower:
        return "WFH Request"
    if "meal subscription" in subject_lower:
        return "Meal Subscription"
    if "mis complaint" in subject_lower:
        return "MIS Complaint"
    return "Unknown Request"


def search_replies_for_employee(employee_id: str):
    """Connects to the inbox and returns all reply emails whose subject contains the
    given employee_id, with request type parsed directly from the subject line."""
    results = []
    try:
        conn = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        conn.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        conn.select("inbox")

        status, data = conn.search(None, f'(SUBJECT "{employee_id}")')
        if status != "OK":
            return results

        for num in data[0].split():
            status, msg_data = conn.fetch(num, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject = _decode(msg.get("Subject"))
            sender = _decode(msg.get("From"))

            if not subject.lower().startswith("re:"):
                continue  # only actual replies

            body = _get_body(msg).strip()
            results.append({
                "subject": subject,
                "from": sender,
                "body": body,
                "request_type": _guess_request_type(subject),
            })

        conn.logout()
    except Exception as e:
        results.append({"error": str(e)})

    return results