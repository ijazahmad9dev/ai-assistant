from datetime import datetime, timedelta
from langchain_core.tools import tool
from app.src.config import GM_EMAIL
from app.tools.email_utils import send_email


@tool
def send_leave_email(
    employee_name: str,
    employee_id: str,
    reason: str,
    days: int,
    start_date: str,
) -> str:
    """Send a leave request email to the GM. ONLY call this AFTER the employee has
    explicitly confirmed (said yes) to a summary you showed them. Requires all fields:
    employee_name, employee_id, reason, days (integer), start_date (format: YYYY-MM-DD).
    The end date is calculated automatically from start_date + days."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = (start + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    except ValueError:
        return "Error: start_date must be in YYYY-MM-DD format."

    subject = f"Leave Request - {employee_name} ({employee_id})"
    body = (
        f"A new leave request has been submitted via the NextBridge chatbot.\n\n"
        f"Employee Name: {employee_name}\n"
        f"Employee ID: {employee_id}\n"
        f"Reason: {reason}\n"
        f"Number of Days: {days}\n"
        f"Start Date: {start_date}\n"
        f"End Date: {end_date}\n"
    )

    result = send_email(GM_EMAIL, subject, body)
    if result == "SUCCESS":
        return f"Leave request sent to the GM for {employee_name} ({start_date} to {end_date})."
    return f"Failed to send leave request: {result}"