from langchain_core.tools import tool
from app.tools.email_utils import send_email
from app.tools.email_reader import search_replies_for_employee


@tool
def send_acknowledgment_email(employee_id: str, request_type: str, message: str) -> str:
    """Send an acknowledgment email back to whoever replied (GM/Food/MIS) for a specific
    request. ONLY call this AFTER the employee has confirmed the drafted acknowledgment
    message. Requires employee_id, request_type (e.g. 'Leave Request'), and message."""
    replies = search_replies_for_employee(employee_id)
    if replies and "error" in replies[0]:
        return f"Could not check inbox: {replies[0]['error']}"

    matches = [r for r in replies if r["request_type"].lower() == request_type.lower()]
    if not matches:
        return f"No reply found for {request_type} to acknowledge."

    reply_to = matches[0]["from"]
    # extract just the email address if "From" includes a display name like "GM <gm@x.com>"
    import re
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", reply_to)
    reply_to_email = email_match.group(0) if email_match else reply_to

    subject = f"Re: Acknowledgment - {request_type}"
    result = send_email(reply_to_email, subject, message)

    if result == "SUCCESS":
        return f"Acknowledgment sent to {reply_to_email} for your {request_type}."
    return f"Failed to send acknowledgment: {result}"