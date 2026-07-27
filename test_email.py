import smtplib
from email.mime.text import MIMEText
from app.src.config import SMTP_SERVER, SMTP_PORT, SMTP_EMAIL, SMTP_APP_PASSWORD, GM_EMAIL

print("SMTP_EMAIL:", repr(SMTP_EMAIL))
print("SMTP_APP_PASSWORD:", repr(SMTP_APP_PASSWORD))
print("GM_EMAIL:", repr(GM_EMAIL))

msg = MIMEText("This is a test email from the NextBridge chatbot setup.")
msg["Subject"] = "Test Email - NextBridge Chatbot"
msg["From"] = SMTP_EMAIL
msg["To"] = GM_EMAIL

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        server.sendmail(SMTP_EMAIL, [GM_EMAIL], msg.as_string())
    print("SUCCESS: Email sent.")
except Exception as e:
    print("FAILED:", e)