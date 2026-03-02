# Topic6VLM — Vision-Language Models (VLM)

This directory contains my work for Topic 6: Vision-Language Models (VLM) in Agentic AI Spring 2026 (CS 6501, University of Virginia).

Course reference:
https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic6VLM/vlm.html


## Overview

This topic focuses on building practical programs using a vision-language model (LLaVA) via Ollama, and structuring VLM-driven applications as agents with clean control flow (LangGraph-style).

Learning goals include:
- Understanding the basic VLM pipeline components (contrastive pretraining encoders, projection alignment, mixed image-language tokens).
- Building an agent that answers questions about uploaded images.
- Handling video input by extracting keyframes and sending them to a VLM.

(See the course page for details.)


## Requirements

Install dependencies (recommended to use a venv/conda env):

pip install -r requirements.txt

You will need:
- Ollama installed
- The llava model pulled
- Python packages:
  - ollama
  - langgraph / langchain (for the LangGraph agent exercise)
  - opencv-python (for the video exercise)

If you do not already have the model, pull it with:

ollama pull llava


## Repository Structure

This folder contains one program per required exercise, plus saved terminal outputs.

Recommended structure:

- exercise1_vlm_langgraph_chat_agent.py
- exercise2_video_surveillance_agent.py
- outputs/
  - exercise1_terminal_output.txt
  - exercise2_terminal_output.txt
- README.md


## Exercise 1 — Vision-Language LangGraph Chat Agent

File:
- exercise1_vlm_langgraph_chat_agent.py

Goal:
- Build a multi-turn chat agent that can hold a conversation about an uploaded image.
- Use good LangGraph structure and manage context carefully.
- If the program runs slowly, reduce the uploaded image resolution.

Implementation notes (from the course page):
- Use Ollama + llava for image Q/A.
- The Ollama chat call supports messages with an "images" key containing file paths and/or base64 strings.

Example model call pattern (reference only; actual implementation is in the script):
- model='llava'
- messages include:
  - role='user'
  - content='Describe this image in English.'
  - images=['./photo.jpg']


## Exercise 2 — Video-Surveillance Agent

File:
- exercise2_video_surveillance_agent.py

Goal:
- Use LLaVA indirectly on video by extracting frames and running a per-frame prompt.
- Create a ~2-minute video clip of an empty space where a person enters and exits at some point.
- Split the video into frames spaced ~2 seconds apart (using OpenCV).
- Run through frames with a prompt asking whether a person is present.
- Report the times at which a person enters and exits the scene.

Install OpenCV:

pip install opencv-python

Core concept:
- Use cv2.VideoCapture("video.mp4")
- Compute FPS
- Sample every ~2 seconds
- Save sampled frames as images
- For each sampled frame, ask LLaVA if a person is present
- Convert frame index to timestamp using FPS and sampling interval


## Running

Exercise 1:

python exercise1_vlm_langgraph_chat_agent.py

Exercise 2:

python exercise2_video_surveillance_agent.py


## Expected Outputs

Exercise 1:
- A multi-turn chat loop where you upload/select an image, then ask multiple questions about it.
- Terminal output saved in outputs/exercise1_terminal_output.txt

Exercise 2:
- Extracted frame images (either in a frames/ folder or current directory).
- Printed timestamps for person entry/exit.
- Terminal output saved in outputs/exercise2_terminal_output.txt


## Resources (from course page)

- Vision-Language Model Guide
- Image Generation Model Guide
- Gradio Quickstart
- tkinter / ipywidgets / gradio UI options

See the Topic 6 course page for the full resource list:
https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic6VLM/vlm.html


## Notes

- This README covers only the required exercises (Exercise 1 and Exercise 2). Optional extensions on the course page are not included here.
- Save terminal logs in outputs/ for portfolio review, as requested by the course page.
