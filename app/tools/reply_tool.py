from langchain_core.tools import tool
from app.tools.email_reader import search_replies_for_employee

# simple in-memory cache so get_reply_details can reuse the last lookup's results
# within the same conversation, without re-hitting IMAP or needing a DB
_last_results_by_employee = {}


@tool
def check_pending_replies(employee_id: str) -> str:
    """Check for replies from GM, Food Department, or MIS for a given employee_id.
    Returns a short list of which requests have replies (type only, no content).
    Call this first when an employee asks for an update. After this, ask which one
    they want details for, then call get_reply_details with that same employee_id
    and the request type they chose."""
    replies = search_replies_for_employee(employee_id)

    if replies and "error" in replies[0]:
        return f"Could not check inbox: {replies[0]['error']}"

    if not replies:
        return f"No replies found for employee ID {employee_id}."

    _last_results_by_employee[employee_id] = replies

    types = sorted(set(r["request_type"] for r in replies))
    lines = [f"- {t}" for t in types]
    return "Replies available for:\n" + "\n".join(lines)


@tool
def get_reply_details(employee_id: str, request_type: str) -> str:
    """Get the full reply content for one specific request type (e.g. 'Leave Request',
    'WFH Request', 'Meal Subscription', 'MIS Complaint'), after the employee has chosen
    which one they want to see from the list check_pending_replies returned. Requires
    employee_id and request_type."""
    replies = _last_results_by_employee.get(employee_id)

    if not replies:
        # fallback: re-fetch if cache is empty (e.g. new session)
        replies = search_replies_for_employee(employee_id)
        if replies and "error" in replies[0]:
            return f"Could not check inbox: {replies[0]['error']}"

    matches = [r for r in replies if r["request_type"].lower() == request_type.lower()]
    if not matches:
        return f"No reply found for {request_type} for employee ID {employee_id}."

    r = matches[0]  # most recent match; could extend to show all if multiple
    return (
        f"Request type: {r['request_type']}\n"
        f"Reply from: {r['from']}\n"
        f"Reply message: {r['body']}\n"
        f"reply_from_email: {r['from']}"
    )