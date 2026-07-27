import smtplib
from email.mime.text import MIMEText
from app.src.config import SMTP_SERVER, SMTP_PORT, SMTP_EMAIL, SMTP_APP_PASSWORD


def send_email(to_email: str, subject: str, body: str) -> str:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_EMAIL, [to_email], msg.as_string())
        return "SUCCESS"
    except Exception as e:
        return f"FAILED: {str(e)}"