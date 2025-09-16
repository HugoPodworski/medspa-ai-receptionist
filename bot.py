#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Twilio + Daily voice bot implementation."""

import os
import sys
import json
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import TTSSpeakFrame, UserStoppedSpeakingFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from ragprocessing import rag_lookup, init_rag_system

from model_config import (
    system_prompt as base_system_prompt,
    tools as function_tools,
    check_availability as mc_check_availability,
    lookup_appointments_for_patient as mc_lookup_appointments_for_patient,
    lookup_patient as mc_lookup_patient,
    create_patient as mc_create_patient,
    book_appointment as mc_book_appointment,
    cancel_appointment as mc_cancel_appointment,
    reschedule_appointment as mc_reschedule_appointment,
    take_message as mc_take_message,
    escalate_to_human as mc_escalate_to_human,
)

# Setup logging
load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Initialize Twilio client
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))


async def run_bot(
    transport: BaseTransport,
    call_id: str,
    sip_uri: str,
    handle_sigint: bool,
    caller_phone=None,
    patient=None,
) -> None:
    """Run the voice bot with the given parameters.

    Args:
        transport: The Daily transport instance
        call_id: The Twilio call ID
        sip_uri: The Daily SIP URI for forwarding the call
    """
    call_already_forwarded = False
    recording_active = False

    # Attempt to initialize RAG; disable gracefully if not configured
    rag_enabled = True
    try:
        await init_rag_system("therapie_clinic_rag")
        logger.info("RAG system initialised")
    except Exception as e:
        rag_enabled = False
        logger.warning(f"RAG disabled (init failed): {e}")


    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(model="qwen-3-235b-a22b-instruct-2507", base_url="https://api.cerebras.ai/v1", api_key=os.getenv("CEREBRAS_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="8d8ce8c9-44a4-46c4-b10f-9a927b99a853",
    )

    current_date_and_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if patient:
        patient_context = f"This patient is recognised in our system.\nName: {patient.get('name')}\nEmail: {patient.get('email')}\nPhone: {patient.get('phone_number')}\nPatient ID: {patient.get('patient_id')}"
        instructions_prompt = "1. Find out what the patient needs help with."
    else:
        patient_context = "No patient record found. Collect phone number, name, and email before proceeding."
        instructions_prompt = "1. Find out what the patient needs help with.\n2. If they are looking to do something that needs a patient record, create one by gathering the phone number, name, and email."

    system_prompt_text = base_system_prompt.format(current_date_and_time=current_date_and_time, instructions_prompt=instructions_prompt, patient_context=patient_context)


    messages = [
        {
            "role": "system",
            "content": system_prompt_text,
        },
    ]

    # Register function-call handlers
    async def handle_check_availability(params: FunctionCallParams):
        try:
            appointment_type = params.arguments.get("appointment_type")
            date = params.arguments.get("date")
            availability = mc_check_availability(appointment_type, date)
            await params.result_callback({
                "appointment_type": appointment_type,
                "date": date,
                "availability": availability,
            })
        except Exception as e:
            await params.result_callback({"error": f"check_availability failed: {str(e)}"})

    async def handle_lookup_appointments_for_patient(params: FunctionCallParams):
        try:
            patient_id = params.arguments.get("patient_id")
            raw = mc_lookup_appointments_for_patient(patient_id)
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                data = {"raw": raw}
            await params.result_callback({
                "patient_id": patient_id,
                "results": data,
            })
        except Exception as e:
            await params.result_callback({"error": f"lookup_appointments_for_patient failed: {str(e)}"})

    async def handle_lookup_patient(params: FunctionCallParams):
        try:
            phone_number = params.arguments.get("phone_number")
            patient = mc_lookup_patient(phone_number)
            await params.result_callback({
                "phone_number": phone_number,
                "patient": patient,
            })
        except Exception as e:
            await params.result_callback({"error": f"lookup_patient failed: {str(e)}"})

    async def handle_create_patient(params: FunctionCallParams):
        try:
            phone_number = params.arguments.get("phone_number")
            name = params.arguments.get("name")
            email = params.arguments.get("email")
            result = mc_create_patient(phone_number, name, email)
            await params.result_callback({
                "message": result,
                "phone_number": phone_number,
                "name": name,
                "email": email,
            })
        except Exception as e:
            await params.result_callback({"error": f"create_patient failed: {str(e)}"})

    async def handle_book_appointment(params: FunctionCallParams):
        try:
            patient_id = params.arguments.get("patient_id")
            appointment_type = params.arguments.get("appointment_type")
            date = params.arguments.get("date")
            result = mc_book_appointment(patient_id, appointment_type, date)
            await params.result_callback({
                "message": result,
                "patient_id": patient_id,
                "appointment_type": appointment_type,
                "date": date,
            })
        except Exception as e:
            await params.result_callback({"error": f"book_appointment failed: {str(e)}"})

    async def handle_cancel_appointment(params: FunctionCallParams):
        try:
            appointment_id = params.arguments.get("appointment_id")
            result = mc_cancel_appointment(appointment_id)
            await params.result_callback({
                "message": result,
                "appointment_id": appointment_id,
            })
        except Exception as e:
            await params.result_callback({"error": f"cancel_appointment failed: {str(e)}"})

    async def handle_reschedule_appointment(params: FunctionCallParams):
        try:
            appointment_id = params.arguments.get("appointment_id")
            new_date = params.arguments.get("new_date")
            result = mc_reschedule_appointment(appointment_id, new_date)
            await params.result_callback({
                "message": result,
                "appointment_id": appointment_id,
                "new_date": new_date,
            })
        except Exception as e:
            await params.result_callback({"error": f"reschedule_appointment failed: {str(e)}"})

    async def handle_take_message(params: FunctionCallParams):
        try:
            message = params.arguments.get("message")
            result = mc_take_message(message)
            await params.result_callback({
                "message": result,
                "user_message": message,
            })
        except Exception as e:
            await params.result_callback({"error": f"take_message failed: {str(e)}"})

    async def handle_escalate_to_human(params: FunctionCallParams):
        try:
            message = params.arguments.get("message")
            result = mc_escalate_to_human(message)
            await params.result_callback({
                "message": result,
                "summary": message,
            })
        except Exception as e:
            await params.result_callback({"error": f"escalate_to_human failed: {str(e)}"})

    # Register the handlers with the LLM
    llm.register_function("check_availability", handle_check_availability)
    llm.register_function("lookup_appointments_for_patient", handle_lookup_appointments_for_patient)
    llm.register_function("lookup_patient", handle_lookup_patient)
    llm.register_function("create_patient", handle_create_patient, cancel_on_interruption=False)
    llm.register_function("book_appointment", handle_book_appointment, cancel_on_interruption=False)
    llm.register_function("cancel_appointment", handle_cancel_appointment, cancel_on_interruption=False)
    llm.register_function("reschedule_appointment", handle_reschedule_appointment, cancel_on_interruption=False)
    llm.register_function("take_message", handle_take_message)
    llm.register_function("escalate_to_human", handle_escalate_to_human)


    # Setup the conversational context
    context = OpenAILLMContext(messages=messages, tools=function_tools)
    context_aggregator = llm.create_context_aggregator(context)
    user_ctx = context_aggregator.user()
    assistant_ctx = context_aggregator.assistant()

    # Helper to extract plain text from message content
    def _extract_text_from_message(message):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts).strip()
        return ""

    # RagProcessor: on user stop, perform RAG and update system prompt before LLM
    class RagProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, UserStoppedSpeakingFrame):
                if rag_enabled:
                    try:
                        # Build a query from the latest assistant and user messages
                        um = [m for m in user_ctx.context.get_messages() if m.get("role") == "user"]
                        am = [m for m in assistant_ctx.context.get_messages() if m.get("role") == "assistant"]
                        last_user = ""
                        for m in reversed(um):
                            last_user = _extract_text_from_message(m)
                            if last_user:
                                break
                        last_assistant = ""
                        for m in reversed(am):
                            last_assistant = _extract_text_from_message(m)
                            if last_assistant:
                                break
                        query = f"Assistant: {last_assistant} User: {last_user}"
                        bullets = await rag_lookup(query)
                        if bullets:
                            messages_current = user_ctx.context.get_messages()
                            sys_index = next((i for i, m in enumerate(messages_current) if m.get("role") == "system"), None)
                            if sys_index is not None:
                                base_content = messages_current[sys_index].get("content", "")
                                marker = "Relevant Context (only use if relevant to the conversation):"
                                if marker in base_content:
                                    base_content = base_content.split(marker)[0].rstrip()
                                new_system_content = f"{base_content}\n\n{marker}\n{bullets}"
                                messages_current[sys_index]["content"] = new_system_content
                                user_ctx.set_messages(messages_current)
                                logger.debug("RAG context updated in system prompt on user stop")
                    except Exception as e:
                        logger.warning(f"RAG update on user stop failed: {e}")

                await self.push_frame(frame, direction)
                return

            # Default: pass through
            await self.push_frame(frame, direction)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_ctx,
            RagProcessor(),   # perform RAG on user stop before LLM
            llm,
            tts,
            transport.output(),
            assistant_ctx,
        ]
    )

    # Create the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    turn_observer = task.turn_tracking_observer

    # No thinking tone hooks

    @turn_observer.event_handler("on_turn_ended")
    async def on_turn_ended(_, turn_number, duration, was_interrupted):
        if was_interrupted:
            logger.info("Barge-in detected on previous turn (was_interrupted=True)")

    # Handle participant joining
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await asyncio.sleep(1.8)
        await task.queue_frames([TTSSpeakFrame(text="Thank you for calling Thérapie Clinic, how can I help you today?")])

    # Handle participant leaving
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        nonlocal recording_active
        if recording_active:
            try:
                await transport.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording on disconnect: {e}")
            recording_active = False
        await task.cancel()

    # Handle call ready to forward
    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, sip_endpoint):
        nonlocal call_already_forwarded

        # We only want to forward the call once
        # The on_dialin_ready event will be triggered for each sip endpoint provisioned
        if call_already_forwarded:
            logger.warning("Call already forwarded, ignoring this event.")
            return

        logger.info(f"Forwarding call {call_id} to {sip_uri}")

        # Retry until Twilio call is in-progress to avoid 21220 redirect error
        max_attempts = 10
        delay_seconds = 0.5
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                twilio_client.calls(call_id).update(
                    twiml=f"<Response><Dial><Sip>{sip_uri}</Sip></Dial></Response>"
                )
                logger.info("Call forwarded successfully")
                call_already_forwarded = True
                break
            except TwilioRestException as e:
                last_error = e
                if getattr(e, "code", None) == 21220:
                    logger.warning(
                        f"Call not in-progress yet (attempt {attempt}/{max_attempts}). Retrying in {delay_seconds}s..."
                    )
                    await asyncio.sleep(delay_seconds)
                    continue
                else:
                    logger.error(f"Failed to forward call: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Failed to forward call: {str(e)}")
                raise
        else:
            logger.error(
                f"Failed to forward call after {max_attempts} attempts. Last error: {str(last_error)}"
            )
            raise last_error if last_error else Exception("Twilio call not in-progress; cannot redirect")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        logger.debug(f"Dial-in connected: {data}")
        nonlocal recording_active
        if not recording_active:
            try:
                await transport.start_recording({})
                recording_active = True
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport, data):
        logger.debug(f"Dial-in stopped: {data}")
        nonlocal recording_active
        if recording_active:
            try:
                await transport.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording on dial-in stopped: {e}")
            recording_active = False

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport, data):
        logger.error(f"Dial-in error: {data}")
        # If there is an error, the bot should leave the call
        # This may be also handled in on_participant_left with
        # await task.cancel()
        nonlocal recording_active
        if recording_active:
            try:
                await transport.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording on dial-in error: {e}")
            recording_active = False

    # Recording status events
    @transport.event_handler("on_recording_started")
    async def on_recording_started(transport, status):
        logger.info(f"Recording started: {status}")

    @transport.event_handler("on_recording_stopped")
    async def on_recording_stopped(transport, status):
        nonlocal recording_active
        recording_active = False
        logger.info(f"Recording stopped: {status}")

    @transport.event_handler("on_recording_error")
    async def on_recording_error(transport, error):
        nonlocal recording_active
        recording_active = False
        logger.error(f"Recording error: {error}")

    @transport.event_handler("on_dialin_warning")
    async def on_dialin_warning(transport, data):
        logger.warning(f"Dial-in warning: {data}")

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    # Extract all details from the body parameter
    body = getattr(runner_args, "body", {})
    room_url = body.get("room_url")
    token = body.get("token")
    call_id = body.get("call_id")
    sip_uri = body.get("sip_uri")
    caller_phone = body.get("caller_phone")
    patient = body.get("patient")
    handle_sigint = body.get("handle_sigint", False)

    if not call_id or not sip_uri:
        logger.error(f"Missing required parameters in body: call_id={call_id}, sip_uri={sip_uri}")
        raise ValueError("call_id and sip_uri are required in the body parameter")

    if not room_url or not token:
        logger.error(f"Missing room connection details: room_url={room_url}, token={token}")
        raise ValueError("room_url and token are required")

    transport = DailyTransport(
        room_url,
        token,
        "Pipecat Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )

    await run_bot(
        transport,
        call_id,
        sip_uri,
        handle_sigint,
        caller_phone=caller_phone,
        patient=patient,
    )