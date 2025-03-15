import logging
import os
from dotenv import load_dotenv
import aiohttp
from typing import Annotated
import re
# LiveKit Agent Imports
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent

# Plugin Imports
from livekit.plugins import openai, silero, turn_detector

# Import the TTS class with Kokoro support
from livekit.plugins.openai.tts import TTS

# Load environment variables
load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)


class AssistantFnc(llm.FunctionContext):
    """
    Defines a set of functions that the assistant can execute.
    """
    pass  # Removed the internet_search function as it was related to Perplexica


def prewarm(proc: JobProcess):
    """
    Runs once before the agent starts to load heavy models (e.g., VAD).
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "你是小K语音助理。您与用户的界面将是语音。"
            "请使用简短而简洁的回答，并避免使用难以发音的标点符号和说一些重复的话。"
            "用中文跟用户去沟通不要使用英文"
        ),
    )

    # Connect to the room and auto-subscribe to audio.
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for at least one participant in the room.
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # 1) Speech-to-Text (STT) with OpenAI + FasterWhisper.
    stt_plugin = openai.STT(
        base_url="http://172.16.200.92:8000/v1",  # Example local endpoint.
        model="Systran/faster-whisper-large-v3",
        api_key="not-needed",
    )

    # 2) Language Model (LLM) from a custom local endpoint.
    llm_plugin = openai.LLM(
        base_url="http://172.16.200.92:11434/v1",  # Example local endpoint.
        api_key="not-needed",  # Your custom API key.
        model="krith/qwen2.5-14b-instruct:IQ1_M",
    )

    # 3) Text-to-Speech (TTS) using Kokoro.
    tts_plugin = TTS.create_kokoro_client(
        model="kokoro",  # Local placeholder model name.
        voice="zf_018",  # Example voice.
        speed=1,
        base_url="http://172.16.200.92:8880/v1",  # Kokoro TTS endpoint.
        api_key="not-needed",  # Typically not needed for local Kokoro.
    )

    # 4) Create the function calling context (previously added internet search functionality).
    fnc_ctx = AssistantFnc()

    # 5) Create the VoicePipelineAgent with the function calling context.
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_ctx,
        turn_detector=turn_detector.EOUModel(),  # End-of-utterance model.
    )

    # Start the agent on the room with the participant.
    agent.start(ctx.room, participant)

    # Greet the participant.
    await agent.say("您好，我是智能语音助手，有什么我能帮助你的吗？", allow_interruptions=True)


if __name__ == "__main__":
    # Run the app with your worker configuration.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )