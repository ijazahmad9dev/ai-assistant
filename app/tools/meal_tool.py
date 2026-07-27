from langchain_core.tools import tool
from app.src.config import FOOD_DEPT_EMAIL
from app.tools.email_utils import send_email

VALID_MEAL_TYPES = {"lunch", "dinner", "both", "roti only"}


@tool
def send_meal_subscription_email(
    employee_name: str,
    employee_id: str,
    meal_type: str,
) -> str:
    """Send a meal subscription request to the food department. ONLY call this AFTER
    the employee has explicitly confirmed (said yes) to a summary you showed them.
    Requires: employee_name, employee_id, meal_type (must be exactly one of:
    'Lunch', 'Dinner', 'Both', 'Roti Only')."""
    if meal_type.strip().lower() not in VALID_MEAL_TYPES:
        return "Error: meal_type must be one of Lunch, Dinner, Both, or Roti Only."

    subject = f"Meal Subscription - {employee_name} ({employee_id})"
    body = (
        f"A new meal subscription request has been submitted via the NextBridge chatbot.\n\n"
        f"Employee Name: {employee_name}\n"
        f"Employee ID: {employee_id}\n"
        f"Meal Subscription: {meal_type}\n"
    )

    result = send_email(FOOD_DEPT_EMAIL, subject, body)
    if result == "SUCCESS":
        return f"Meal subscription ({meal_type}) sent to the Food Department for {employee_name}."
    return f"Failed to send meal subscription: {result}"