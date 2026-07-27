from langchain_core.tools import tool
from app.src.config import MIS_EMAIL
from app.tools.email_utils import send_email


@tool
def send_mis_complaint_email(
    employee_name: str,
    employee_id: str,
    issue_description: str,
) -> str:
    """Send a hardware/system/operational issue complaint to the MIS department. ONLY
    call this AFTER the employee has explicitly confirmed (said yes) to a summary you
    showed them. Requires: employee_name, employee_id, issue_description (details of
    the hardware or operational problem)."""
    subject = f"MIS Complaint - {employee_name} ({employee_id})"
    body = (
        f"A new MIS complaint has been submitted via the NextBridge chatbot.\n\n"
        f"Employee Name: {employee_name}\n"
        f"Employee ID: {employee_id}\n"
        f"Issue Description: {issue_description}\n"
    )

    result = send_email(MIS_EMAIL, subject, body)
    if result == "SUCCESS":
        return f"Complaint submitted to MIS for {employee_name}."
    return f"Failed to send MIS complaint: {result}"