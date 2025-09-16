from datetime import datetime
import uuid
import json

system_prompt="""
<role>
You are a receptionist for Thérapie Clinic. You are responsible for helping patients with their appointments as well as answering their questions.
</role>

<context>
- You are on a phone call with the user.
- All the users inputs are coming from a phone call and then being transcribed and therefore sometimes they are not very clear, so you need to either clarify or assume the correct answer.
- All your outputs are being spoken out loud via a TTS model so keep your responses conversational for example you should say "1st" not "1." or "2nd" not "2."
- Current date and time: {current_date_and_time}
</context>

<style>
- Your responses are concise and to the point, ideally in 10-15 words max.
- Your responses are natural and conversational.
- You don't repeat yourself.
</style>

<patient_context>
{patient_context}
</patient_context>

<instructions>
{instructions_prompt}
</instructions>

<examples>
1. Rescheduling an appointment:
User: "Hello I was looking to move my appointment to next week."
Assistant: "Your number wasn't recognised in our system, could you provide the number you used to book the appointment?"
User: "1 4 1 5 5 5 5 0 1 9 8"
Assistant: "Ok thank you give me one second to search this up"
Assistant: *uses lookup_patient tool*
Assistant: *uses lookup_appointments_for_patient tool*
Assistant: "Ok thank you I see that you have an appointment scheduled for this Tuesday at 10:00 AM, is that the one you want to reschedule?"
User: "Yes that's the one, can you reschedule it to next week?"
Assistant: "To Tuesday 10am next week correct?"
User: "Yes that's correct"
Assistant: *uses reschedule_appointment tool*
Assistant: "Your appointment has been rescheduled to Tuesday next week, which is the 25th of September at 10:00 AM, is there anything else I can help you with?"
User: "No that's all thank you very much"
Assistant: "You're welcome, have a great day!"

2. Booking an appointment:
User: "I was looking to book some laser hair removal for my legs"
Assistant: "Is that the full legs or just the lower half?"
User: "Just the lower half"
Assistant: "Ok and when would you like to come in?"
User: "Next week on Tuesday"
Assistant: *uses check_availability tool*
Assistant: "We have 10am available and 3pm available, do any of these work for you?"
User: "Yes 3pm works for me"
Assistant: *uses book_appointment tool*
Assistant: "Your appointment has been booked for Tuesday next week, which is the 25th of September at 3:00 PM, is there anything else I can help you with?"
User: "No that's all thank you very much"
Assistant: "You're welcome, have a great day!"
</examples>

<notes>
- Always use the tools provided to you to answer the user's question.
- If you don't have the information to answer the users question, just let them know that you don't have the information and escalate to the human.
- When calling tools, provide only strict JSON arguments matching the schema. Do not include code fences or any extra characters.
</notes>
"""

_PATIENTS_BY_ID: dict[str, dict] = {
    "pt_123456": {
        "patient_id": "pt_123456",
        "name": "Peter Parker",
        "email": "peter.parker@example.com",
        "phone_number": "",
        "created_at": "2025-01-01T00:00:00Z",
    },
    "pt_10293a": {
        "patient_id": "pt_10293a",
        "name": "Lauren Park",
        "email": "lauren.park@example.com",
        "phone_number": "+14155550198",
        "created_at": "2025-01-01T00:00:00Z",
    },
    "pt_98ff31": {
        "patient_id": "pt_98ff31",
        "name": "Miguel Alvarez",
        "email": "miguel.alvarez@example.com",
        "phone_number": "+15125550140",
        "created_at": "2025-01-01T00:00:00Z",
    },
    "pt_75b2d9": {
        "patient_id": "pt_75b2d9",
        "name": "Priya Nair",
        "email": "priya.nair@example.com",
        "phone_number": "+16175550127",
        "created_at": "2025-01-01T00:00:00Z",
    },
}

APPOINTMENT_TYPE_ENUM = [
    # Core types
    "consultation_virtual",
    "consultation_physical",
    "follow_up",
    "service",
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if appointments are available for a given appointment type on a specific date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_type": {
                        "type": "string",
                        "description": "The type of appointment you are checking availability for.",
                        "enum": APPOINTMENT_TYPE_ENUM,
                    },
                    "date": {
                        "type": "string",
                        "description": "Target date in YYYY-MM-DD."
                    }
                },
                "required": ["appointment_type", "date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_appointments_for_patient",
            "description": "List appointments for a patient by their patient_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Unique identifier for the patient."
                    }
                },
                "required": ["patient_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for the patient on the specified date. Ensure required fields are present and check availability when appropriate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Unique identifier for the patient."
                    },
                    "appointment_type": {
                        "type": "string",
                        "description": "The type of appointment you are booking.",
                        "enum": APPOINTMENT_TYPE_ENUM,
                    },
                    "date": {
                        "type": "string",
                        "description": "Appointment date in YYYY-MM-DD."
                    }
                },
                "required": ["patient_id", "appointment_type", "date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel a scheduled appointment by its appointment_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "Unique identifier of the appointment to cancel."
                    }
                },
                "required": ["appointment_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an existing appointment to a new date. Check availability when appropriate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "Unique identifier of the appointment to reschedule."
                    },
                    "new_date": {
                        "type": "string",
                        "description": "New appointment date in YYYY-MM-DD."
                    }
                },
                "required": ["appointment_id", "new_date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_message",
            "description": "Record a message for clinic staff when the user wants to leave information or a request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The user’s message for clinic staff."
                    }
                },
                "required": ["message"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the request to human staff. Include a concise summary of the situation and what is needed to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Summary of context, what the user wants, and any missing info."
                    }
                },
                "required": ["message"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_patient",
            "description": "Find a patient by their phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "Phone number of the patient."
                    }
                },
                "required": ["phone_number"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_patient",
            "description": "Create a new patient with phone number, name, and email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "Phone number of the patient."
                    },
                    "name": {
                        "type": "string",
                        "description": "Full name of the patient."
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address of the patient."
                    }
                },
                "required": ["phone_number", "name", "email"],
                "additionalProperties": False
            }
        }
    }
]

def check_availability(appointment_type: str, date: str):
    
    return "Available"

def lookup_appointments_for_patient(patient_id: str):
    dummy_appointment = {
        "appointment_id": "appt_12345",
        "appointment_date": "2025-09-10",
        "appointment_time": "14:30",
        "name": "Consultation with Dr. Smith"
    }
    return json.dumps({"appointments": [dummy_appointment]})

def lookup_patient(phone_number: str):
    for patient in _PATIENTS_BY_ID.values():
        if patient["phone_number"] == phone_number:
            return patient
        
def create_patient(phone_number: str, name: str, email: str):
    new_id = uuid.uuid4().hex[:6]
    _PATIENTS_BY_ID[new_id] = {"patient_id": new_id, "name": name, "email": email, "phone_number": phone_number, "created_at": datetime.now().isoformat()}
    return f"Patient created successfully. Patient ID: {new_id}"

def book_appointment(patient_id: str, appointment_type: str, date: str):
    return f"Appointment booked successfully. Appointment ID: {uuid.uuid4().hex[:8]}"

def cancel_appointment(appointment_id: str):
    return "Appointment cancelled successfully."

def reschedule_appointment(appointment_id: str, new_date: str):
    return "Appointment rescheduled successfully."

def take_message(message: str):
    return "Message taken successfully."

def transfer_to_human(message: str):
    return "Transferring to human staff."