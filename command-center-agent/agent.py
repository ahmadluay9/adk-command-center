import os
from callback_logging import log_query_to_model, log_model_response

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from google.genai import types
from typing import Optional, Dict, Any, List, TypedDict
from dotenv import load_dotenv

load_dotenv()
model_name = os.getenv("MODEL")
os.environ['GOOGLE_CLOUD_PROJECT'] = os.getenv('GOOGLE_CLOUD_PROJECT')
os.environ['GOOGLE_CLOUD_LOCATION'] = os.getenv('GOOGLE_CLOUD_LOCATION')
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')

# Healthcare tools for the command center agent
# These tools simulate interactions with a hospital system, allowing the agent
# to admit patients, find available beds, assign doctors, and retrieve patient
# statuses. They are designed to work with the session state, which acts as a
# mock database for patients and beds.

def admit_patient(patient_name: str, condition: str, tool_context: ToolContext) -> dict:
    """
    Admits a new patient into the hospital system. Creates a record for them.

    Args:
        patient_name (str): The full name of the patient.
        condition (str): The primary medical condition of the patient upon admission.
        tool_context (ToolContext): Gives the tool access to session state.

    Returns:
        dict: A dictionary confirming the admission or stating an error.
    """
    print(f"--- Tool: admit_patient called for {patient_name} ---")
    state = tool_context.state
    # Make a copy of the patients dictionary to ensure changes are tracked
    patients_db = state.get("patients", {}).copy()

    if patient_name in patients_db:
        return {"status": "error", "message": f"Patient '{patient_name}' is already admitted."}

    # Add the new patient to our copied dictionary
    patients_db[patient_name] = {
        "condition": condition,
        "bed": None,
        "doctor": None,
        "status": "Admitted"
    }
    # Re-assign the entire updated dictionary back to the state
    state["patients"] = patients_db
    print(f"--- Tool: Updated state with new patient: {patient_name} ---")
    return {"status": "success", "message": f"Patient '{patient_name}' admitted successfully."}

def find_available_bed(patient_name: str, department: str, tool_context: ToolContext) -> dict:
    """
    Finds and assigns an available bed to a patient in a specific department.

    Args:
        patient_name (str): The name of the patient needing a bed.
        department (str): The hospital department (e.g., 'Cardiology', 'ICU').
        tool_context (ToolContext): Gives the tool access to session state.

    Returns:
        dict: A dictionary confirming the bed assignment or stating an error.
    """
    print(f"--- Tool: find_available_bed called for {patient_name} in {department} ---")
    state = tool_context.state
    # Make copies of the state dictionaries to ensure changes are tracked
    patients_db = state.get("patients", {}).copy()
    beds_db = state.get("beds", {}).copy()

    if patient_name not in patients_db:
        return {"status": "error", "message": f"Patient '{patient_name}' not found."}

    # Find a bed in the requested department that is not occupied
    available_bed = None
    # Ensure department exists before iterating
    if department in beds_db:
      for bed_id, bed_info in beds_db[department].items():
          if bed_info["occupied"] is None:
              available_bed = bed_id
              break

    if available_bed:
        # Modify the copies
        beds_db[department][available_bed]["occupied"] = patient_name
        patients_db[patient_name]["bed"] = f"{department} - {available_bed}"
        patients_db[patient_name]["status"] = "Bed Assigned"
        # Re-assign the updated dictionaries back to the state
        state["beds"] = beds_db
        state["patients"] = patients_db
        print(f"--- Tool: Assigned {patient_name} to bed {available_bed} ---")
        return {"status": "success", "message": f"Assigned {patient_name} to bed {available_bed} in {department}."}
    else:
        return {"status": "error", "message": f"No available beds in {department}."}

def assign_doctor(patient_name: str, tool_context: ToolContext) -> dict:
    """
    Assigns an available doctor to a patient based on their condition.

    Args:
        patient_name (str): The name of the patient needing a doctor.
        tool_context (ToolContext): Gives the tool access to session state.

    Returns:
        dict: A dictionary confirming the doctor assignment or stating an error.
    """
    print(f"--- Tool: assign_doctor called for {patient_name} ---")
    state = tool_context.state
    # Make a copy of the patients dictionary to ensure changes are tracked
    patients_db = state.get("patients", {}).copy()

    if patient_name not in patients_db:
        return {"status": "error", "message": f"Patient '{patient_name}' not found."}

    # Mock doctor assignment logic
    patient_condition = patients_db[patient_name].get("condition", "").lower()
    doctor = "Dr. Smith (Cardiologist)" if "cardiac" in patient_condition else "Dr. Jones (General)"

    # Modify the copy
    patients_db[patient_name]["doctor"] = doctor
    patients_db[patient_name]["status"] = "Doctor Assigned"
    # Re-assign the updated dictionary back to the state
    state["patients"] = patients_db
    print(f"--- Tool: Assigned {doctor} to {patient_name} ---")
    return {"status": "success", "message": f"Assigned {doctor} to {patient_name}."}


def get_patient_status(patient_name: str, tool_context: ToolContext) -> dict:
    """
    Retrieves the complete current status of a specific patient.

    Args:
        patient_name (str): The name of the patient to check.
        tool_context (ToolContext): Gives the tool access to session state.

    Returns:
        dict: A dictionary with the patient's status or an error message.
    """
    print(f"--- Tool: get_patient_status called for {patient_name} ---")
    state = tool_context.state
    patient_info = state.get("patients", {}).get(patient_name)

    if patient_info:
        return {"status": "success", "data": patient_info}
    else:
        return {"status": "error", "message": f"Patient '{patient_name}' not found."}

# This dictionary represents the initial state of our mock hospital database.
initial_hospital_state = {
    "beds": {
        "Cardiology": {"C-101": {"occupied": None}, "C-102": {"occupied": None}},
        "ICU": {"ICU-A": {"occupied": None}, "ICU-B": {"occupied": "Jane Smith"}},
    },
    "patients": {}
}

# This callback runs at the start of every new session to set up the 'database'.
def initialize_state_callback(callback_context: CallbackContext) -> None:
    """Checks if the session state is empty and initializes it."""
    if not callback_context.state:
        print("--- Callback: State is empty. Initializing with hospital data. ---")
        callback_context.state.update(initial_hospital_state)


# Specialized Sub-Agents
intake_agent = Agent(
    model=model_name, name="intake_agent",
    instruction="You are the Intake Agent. Your ONLY task is to admit new patients into the system using the 'admit_patient' tool.",
    description="Handles the admission of new patients into the hospital.",
    tools=[admit_patient],
)

bed_management_agent = Agent(
    model=model_name, name="bed_management_agent",
    instruction="You are the Bed Management Agent. Your ONLY task is to find and assign beds to patients using the 'find_available_bed' tool.",
    description="Finds and assigns available hospital beds to patients.",
    tools=[find_available_bed],
)

doctor_assignment_agent = Agent(
    model=model_name, name="doctor_assignment_agent",
    instruction="You are the Doctor Assignment Agent. Your ONLY task is to assign a doctor to a patient using the 'assign_doctor' tool.",
    description="Assigns an available doctor to a patient.",
    tools=[assign_doctor],
)

# Root "Command Center" Agent
root_agent = Agent(
    name="command_center_agent", model=model_name,
    description="The main coordinator for hospital operations. It can check patient status and delegate tasks.",
    instruction=(
        "You are the Hospital Command Center coordinator. Your primary role is to orchestrate patient care. "
        "1. If a request is about admitting a new patient, delegate to 'intake_agent'. "
        "2. If a request is about finding a bed, delegate to 'bed_management_agent'. "
        "3. If a request is about assigning a doctor, delegate to 'doctor_assignment_agent'. "
        "4. If a user asks for a patient's status, handle it yourself using the 'get_patient_status' tool. "
        "Analyze the user's request and delegate to the correct specialist agent."
    ),
    tools=[get_patient_status],
    sub_agents=[intake_agent, bed_management_agent, doctor_assignment_agent]
)

# import os
# import sys
# sys.path.append("..")
# from callback_logging import log_query_to_model, log_model_response

# from dotenv import load_dotenv
# from datetime import datetime, timedelta
# import dateparser

# from google.adk import Agent

# load_dotenv()
# model_name = os.getenv("MODEL")


# def get_date(x_days_from_today:int):
#     """
#     Retrieves a date for today or a day relative to today.

#     Args:
#         x_days_from_today (int): how many days from today? (use 0 for today)

#     Returns:
#         A dict with the date in a formal writing format. For example:
#         {"date": "Wednesday, May 7, 2025"}
#     """

#     target_date = datetime.today() + timedelta(days=x_days_from_today)
#     date_string = target_date.strftime("%A, %B %d, %Y")

#     return {"date": date_string}

# def write_journal_entry(entry_date:str, journal_content:str):
#     """
#     Writes a journal entry based on the user's thoughts.

#     Args:
#         entry_date (str): The entry date of the journal entry)
#         journal_content (str): The body text of the journal entry

#     Returns:
#         A dict with the filename of the written entry. For example:
#         {"entry": "2025-05-07.txt"}
#         Or a dict indicating an error, For example:
#         {"status": "error"}
#     """

#     date_for_filename = dateparser.parse(entry_date).strftime("%Y-%m-%d")
#     filename = f"{date_for_filename}.txt"
    
#     # Create the file if it doesn't already exist
#     if not os.path.exists(filename):
#         print(f"Creating a new journal entry: {filename}")
#         with open(filename, "w") as f:
#             f.write("### " + entry_date)

#     # Append to the dated entry
#     try:
#         with open(filename, "a") as f:
#             f.write("\n\n" + journal_content)            
#         return {"entry": filename}
#     except:
#         return {"status": "error"}

# root_agent = Agent(
#     name="function_tool_agent",
#     model=model_name,
#     description="Help users practice good daily journalling habits.",
#     instruction="""
#     Ask the user how their day is going and
#     use their response to write a journal entry for them.""",
#     before_model_callback=log_query_to_model,
#     after_model_callback=log_model_response,
#     # Add the function tools below
#     tools=[get_date, write_journal_entry]

# )