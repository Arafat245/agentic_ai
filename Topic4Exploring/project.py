"""
Simple 2-hour project: YouTube Transcript ReAct agent.

What this script does:
1. Accepts a YouTube URL or raw video ID
2. Uses a custom tool to fetch the transcript
3. Lets a LangGraph ReAct agent summarize or answer questions

Requirements:
    pip install langgraph langchain-openai langchain-core youtube-transcript-api

Environment:
    export OPENAI_API_KEY="your-key-here"

Run:
    python project.py
"""

from __future__ import annotations

import os
from typing import List
from urllib.parse import parse_qs, urlparse

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(video_input: str) -> str:
    """Return the 11-character YouTube video ID from a URL or raw ID."""
    candidate = video_input.strip()

    # Raw YouTube IDs are typically 11 characters.
    if len(candidate) == 11 and "/" not in candidate and "?" not in candidate:
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if "youtube.com" in host or "m.youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v", [])
        if query_id:
            return query_id[0]

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "live", "embed"}:
            return path_parts[1]

    raise ValueError(
        "Could not extract a YouTube video ID. "
        "Please provide a full YouTube URL or the raw 11-character video ID."
    )


@tool
def get_youtube_transcript(video_input: str) -> str:
    """
    Fetch a YouTube transcript from a full URL or raw video ID.

    Use this tool whenever the user asks to summarize a YouTube video,
    explain a video, answer questions about a video, or produce notes
    from YouTube content.
    """
    video_id = extract_video_id(video_input)

    try:
        transcript_obj = YouTubeTranscriptApi().fetch(video_id)
        chunks = [snippet.text for snippet in transcript_obj]
    except Exception:
        # Fallback for installations that still expose the older API.
        transcript_obj = YouTubeTranscriptApi.get_transcript(video_id)
        chunks = [snippet["text"] for snippet in transcript_obj]

    full_text = " ".join(chunk.strip() for chunk in chunks if chunk.strip())

    if not full_text:
        return f"No transcript text was found for video `{video_id}`."

    max_chars = 12000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + (
            "\n\n[Transcript truncated to keep the prompt small.]"
        )

    return f"Video ID: {video_id}\nTranscript:\n{full_text}"


def build_agent():
    """Create a minimal ReAct agent with one custom transcript tool."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it in your terminal first."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(model=model_name, temperature=0)

    system_prompt = (
        "You are a helpful study assistant for YouTube videos. "
        "When the user asks about a YouTube video, always use the "
        "get_youtube_transcript tool first. Then summarize clearly, "
        "answer questions accurately, and mention if the transcript was truncated."
    )

    return create_react_agent(
        model=model,
        tools=[get_youtube_transcript],
        prompt=system_prompt,
    )


def print_intro() -> None:
    """Show a short startup guide."""
    print("=" * 80)
    print("YouTube Transcript ReAct Agent")
    print("=" * 80)
    print("Type a normal question and include a YouTube URL or video ID.")
    print("Type 'exit' or 'quit' to stop.")
    print("\nExample prompts:")
    print("  - Summarize this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("  - Give me 5 bullet points from video dQw4w9WgXcQ")
    print("  - What are the key ideas in this video: https://youtu.be/dQw4w9WgXcQ")
    print("=" * 80)


def chat() -> None:
    """Run a simple multi-turn terminal chat."""
    agent = build_agent()
    messages: List = []

    print_intro()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\nSession ended.")
            break

        messages.append(("user", user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        final_message = messages[-1]
        print(f"\nAssistant: {final_message.content}")


if __name__ == "__main__":
    chat()
