#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Webhook server to handle Twilio calls and start the voice bot.

This server provides two main endpoints:
- /call: Twilio webhook handler that receives incoming calls
- /start: Bot starting endpoint for local development (mimics Pipecat Cloud)

The server automatically detects the environment (local vs production) and routes
bot starting requests accordingly:
- Local: Uses internal /start endpoint
- Production: Calls Pipecat Cloud API

All call data (room_url, token, call_id, sip_uri) flows through the body parameter
to ensure consistency between local and cloud deployments.
"""

import os
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from loguru import logger
from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)
from twilio.twiml.voice_response import VoiceResponse
from ragprocessing import init_rag_system, shutdown_rag
from model_config import lookup_patient

# Load environment variables
load_dotenv()


# Initialize FastAPI app with aiohttp session
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create aiohttp session to be used for Daily API calls
    app.state.session = aiohttp.ClientSession()
    # Initialize shared RAG resources once at startup
    await init_rag_system(os.getenv("RAG_COLLECTION_NAME", "therapie_clinic_rag"))
    logger.info("RAG system initialised")
    yield
    # Close session when shutting down
    await app.state.session.close()
    # Shutdown RAG resources
    await shutdown_rag()


app = FastAPI(lifespan=lifespan)


class SipRoomConfig:
    def __init__(self, room_url: str, token: str, sip_endpoint: Optional[str] = None):
        self.room_url = room_url
        self.token = token
        self.sip_endpoint = sip_endpoint


async def create_sip_room_without_dialout(
    aiohttp_session: aiohttp.ClientSession,
    *,
    sip_caller_phone: str,
    room_exp_duration: float = 2.0,
    token_exp_duration: float = 2.0,
    sip_enable_video: bool = False,
    sip_num_endpoints: int = 1,
    sip_codecs: Optional[dict] = None,
) -> SipRoomConfig:
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise Exception(
            "DAILY_API_KEY environment variable is required. Get your API key from https://dashboard.daily.co/developers"
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    room_name = f"pipecat-sip-{uuid.uuid4().hex[:8]}"
    logger.info(f"Creating new Daily room: {room_name}")

    expiration_time = time.time() + (room_exp_duration * 60 * 60)

    room_properties = DailyRoomProperties(
        exp=expiration_time,
        eject_at_room_exp=True,
        start_video_off=not sip_enable_video,
        enable_recording="cloud",
    )

    # Configure SIP dial-in without enabling dial-out
    sip_params = DailyRoomSipParams(
        display_name=sip_caller_phone,
        video=sip_enable_video,
        sip_mode="dial-in",
        num_endpoints=sip_num_endpoints,
        codecs=sip_codecs,
    )
    room_properties.sip = sip_params
    # Do NOT set room_properties.enable_dialout here; default is False/omitted

    room_params = DailyRoomParams(name=room_name, properties=room_properties)

    room_response = await daily_rest_helper.create_room(room_params)
    room_url = room_response.url
    logger.info(f"Created Daily room: {room_url}")

    token_expiry_seconds = token_exp_duration * 60 * 60
    token = await daily_rest_helper.get_token(room_url, token_expiry_seconds)

    sip_endpoint = room_response.config.sip_endpoint
    logger.info(f"SIP endpoint: {sip_endpoint}")

    return SipRoomConfig(room_url=room_url, token=token, sip_endpoint=sip_endpoint)


@app.post("/call", response_class=PlainTextResponse)
async def handle_call(request: Request):
    """Handle incoming Twilio call webhook.

    This endpoint:
    1. Receives Twilio webhook data for incoming calls
    2. Creates a Daily room with SIP capabilities
    3. Starts the bot (locally or via Pipecat Cloud based on ENVIRONMENT)
    4. Returns TwiML to put caller on hold while bot connects

    Returns:
        TwiML response with hold music for the caller
    """
    logger.debug("Received call webhook from Twilio")

    try:
        # Get form data from Twilio webhook
        form_data = await request.form()
        data = dict(form_data)

        # Extract call ID (required to forward the call later)
        call_sid = data.get("CallSid")
        if not call_sid:
            raise HTTPException(status_code=400, detail="Missing CallSid in request")

        # Extract the caller's phone number
        caller_phone = str(data.get("From", "unknown-caller"))
        logger.debug(f"Processing call with ID: {call_sid} from {caller_phone}")

        # Attempt to look up the patient by phone number right away
        try:
            patient = lookup_patient(caller_phone)
            if patient:
                logger.debug(f"Matched patient by phone: {patient.get('patient_id')} - {patient.get('name')}")
            else:
                logger.debug("No patient match found for caller phone")
        except Exception as e:
            logger.warning(f"Patient lookup failed: {e}")
            patient = None

        # Create a Daily room with SIP capabilities, without dial-out enabled
        try:
            sip_config = await create_sip_room_without_dialout(
                request.app.state.session, sip_caller_phone=caller_phone
            )
        except Exception as e:
            logger.error(f"Error creating Daily room: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

        # Extract necessary details
        room_url = sip_config.room_url
        token = sip_config.token
        sip_endpoint = sip_config.sip_endpoint

        # Make sure we have a SIP endpoint
        if not sip_endpoint:
            raise HTTPException(status_code=500, detail="No SIP endpoint provided by Daily")

        # Start the bot - either locally or via Pipecat Cloud
        try:
            # Check environment mode (local development vs production)
            environment = os.getenv("ENVIRONMENT", "local")  # "local" or "production"

            # Prepare body data with all necessary information
            # This data structure is consistent between local and cloud deployments
            body_data = {
                "room_url": room_url,
                "token": token,
                "call_id": call_sid,
                "sip_uri": sip_endpoint,
                "caller_phone": caller_phone,
                "patient": patient,
            }

            if environment == "production":
                # Production: Call Pipecat Cloud API to start the bot
                pipecat_api_token = os.getenv("PIPECAT_API_TOKEN")
                agent_name = os.getenv("PIPECAT_AGENT_NAME")

                if not pipecat_api_token:
                    raise HTTPException(
                        status_code=500, detail="PIPECAT_API_TOKEN required for production mode"
                    )

                logger.debug(f"Starting bot via Pipecat Cloud for call {call_sid}")
                async with request.app.state.session.post(
                    f"https://api.pipecat.daily.co/v1/public/{agent_name}/start",
                    headers={
                        "Authorization": f"Bearer {pipecat_api_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "createDailyRoom": False,  # We already created the room
                        "body": body_data,
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to start bot via Pipecat Cloud: {error_text}",
                        )
                    cloud_data = await response.json()
                    logger.debug(f"Bot started successfully via Pipecat Cloud")
            else:
                # Local development: Call internal /start endpoint to start the bot
                local_server_url = os.getenv("LOCAL_SERVER_URL", "http://localhost:7860")

                logger.debug(f"Starting bot via local /start endpoint for call {call_sid}")
                async with request.app.state.session.post(
                    f"{local_server_url}/start",
                    headers={"Content-Type": "application/json"},
                    json={
                        "createDailyRoom": False,  # We already created the room
                        "body": body_data,
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to start bot via local /start endpoint: {error_text}",
                        )
                    local_data = await response.json()
                    logger.debug(f"Bot started successfully via local /start endpoint")

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

        # Generate TwiML response to put the caller on hold with music
        # The caller hears this while the bot connects to the Daily room
        # You can replace the URL with your own music file or use Twilio's built-in music
        # See: https://www.twilio.com/docs/voice/twiml/play#music-on-hold
        resp = VoiceResponse()
        resp.play(
            url="https://therapeutic-crayon-2467.twil.io/assets/US_ringback_tone.mp3",
            loop=10,
        )

        return str(resp)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/start")
async def start_bot_endpoint(request: Request):
    """Start bot endpoint for local development.

    This endpoint mimics the Pipecat Cloud API pattern, receiving the same body data
    structure and starting the bot locally. Used only in local development mode.

    Args:
        request: FastAPI request containing body with room_url, token, call_id, sip_uri

    Returns:
        dict: Success status and call_id
    """
    try:
        # Parse the request body
        request_data = await request.json()
        body = request_data.get("body", {})

        # Extract required data from body
        room_url = body.get("room_url")
        token = body.get("token")
        call_id = body.get("call_id")
        sip_uri = body.get("sip_uri")

        if not all([room_url, token, call_id, sip_uri]):
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters in body: room_url, token, call_id, sip_uri",
            )

        from pipecat.runner.types import DailyRunnerArguments

        from bot import bot as bot_function

        # Create runner arguments with body data
        # Note: room_url and token are passed via body, not as direct arguments
        runner_args = DailyRunnerArguments(
            room_url=None,  # Data comes from body
            token=None,  # Data comes from body
            body=body,
        )
        runner_args.handle_sigint = False

        # Start the bot in the background
        import asyncio

        asyncio.create_task(bot_function(runner_args))

        return {"status": "Bot started successfully", "call_id": call_id}

    except Exception as e:
        logger.error(f"Error in /start endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Status indicating server health
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "7860"))
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
