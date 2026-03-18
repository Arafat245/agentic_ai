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
```bash
pip install -r requirements.txt
```

You will need:
- Ollama installed
- The llava model pulled
- Python packages:
  - ollama
  - langgraph / langchain (for the LangGraph agent exercise)
  - opencv-python (for the video exercise)

If you do not already have the model, pull it with:
```bash
ollama pull llava
```


## Repository Structure

This folder contains one program per required exercise, plus saved terminal outputs, a screenshot, and extracted video frames.

```
Topic6VLM/
├── exercise1_vlm_langgraph_chat_agent.py
├── Exercise2.py                    # Video surveillance agent
├── 2min.mp4                         # Test video for Exercise 2
├── requirements.txt
├── outputs/
│   ├── exercise1_terminal_output.txt
│   ├── exercise2_terminal_output.txt
│   └── screenshot.png              # Screenshot of Exercise 1 Gradio interface
├── frames/                          # Extracted frames from Exercise 2 (frame_0000.jpg, ...)
└── README.md
```


## Exercise 1 — Vision-Language LangGraph Chat Agent

File:
- exercise1_vlm_langgraph_chat_agent.py

Goal:
- Build a multi-turn chat agent that can hold a conversation about an uploaded image.
- Use good LangGraph structure and manage context carefully.
- If the program runs slowly, reduce the uploaded image resolution.

Implementation:
- **Gradio interface** (default): Web UI with image upload and chat. Run `python3 exercise1_vlm_langgraph_chat_agent.py`
- **CLI interface**: Text-based flow. Run `python3 exercise1_vlm_langgraph_chat_agent.py --cli`
- Uses Ollama + LLaVA (`llava:latest`) for vision-language inference
- LangGraph pipeline: add_user_message → call_vlm → trim_history
- Image handling: loads from file path immediately, clears conversation on new image upload

### Screenshot

![Exercise 1 — VLM LangGraph Chat Agent](outputs/screenshot.png)


## Exercise 2 — Video-Surveillance Agent

File:
- Exercise2.py

Goal:
- Use LLaVA indirectly on video by extracting frames and running a per-frame prompt.
- Create a ~2-minute video clip of an empty space where a person enters and exits at some point.
- Split the video into frames spaced ~2 seconds apart (using OpenCV).
- Run through frames with a prompt asking whether a person is present.
- Report the times at which a person enters and exits the scene.

Implementation:
- **Video**: `2min.mp4` (2-minute test clip)
- **Frame extraction**: OpenCV, one frame every 2 seconds (~63 frames for 2 min)
- **Person detection**: LLaVA (`llava:latest`) with JSON prompt per frame
- **Output**: Frames saved to `frames/`, ENTRY/EXIT timestamps printed


## Running

**Prerequisites:** Start Ollama and pull the model:
```bash
ollama serve          # In a separate terminal
ollama pull llava
```

Exercise 1 (Gradio):
```bash
python3 exercise1_vlm_langgraph_chat_agent.py
```

Exercise 1 (CLI):
```bash
python3 exercise1_vlm_langgraph_chat_agent.py --cli
```

Exercise 2:
```bash
python3 Exercise2.py
```

**Saving terminal output** (use `-u` for unbuffered output so logs are captured):
```bash
python3 -u exercise1_vlm_langgraph_chat_agent.py 2>&1 | tee outputs/exercise1_terminal_output.txt
python3 -u Exercise2.py 2>&1 | tee outputs/exercise2_terminal_output.txt
```


## Outputs

### Exercise 1
- **Terminal output**: [outputs/exercise1_terminal_output.txt](outputs/exercise1_terminal_output.txt)
- **Screenshot**: [outputs/screenshot.png](outputs/screenshot.png)

Sample (Gradio startup + VLM calls):
```
Launching Gradio interface...
* Running on local URL:  http://127.0.0.1:7860
[VLM] Sending 2 messages, image attached: True
...
```

### Exercise 2
- **Terminal output**: [outputs/exercise2_terminal_output.txt](outputs/exercise2_terminal_output.txt)
- **Extracted frames**: `frames/frame_0000.jpg` through `frame_0062.jpg`

Sample (Surveillance Report):
```
============================================================
Surveillance Report
============================================================
  ENTRY at 00:07
  EXIT at 00:09
  ENTRY at 00:13
  EXIT at 00:17
  ...
Done.
```


## Resources (from course page)

- Vision-Language Model Guide
- Image Generation Model Guide
- Gradio Quickstart
- tkinter / ipywidgets / gradio UI options

See the Topic 6 course page for the full resource list:
https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic6VLM/vlm.html


## Notes

- This README covers only the required exercises (Exercise 1 and Exercise 2). Optional extensions on the course page are not included here.
- Terminal logs are saved in `outputs/` for portfolio review, as requested by the course page.
- Use `python3` (not `python`) on systems where only `python3` is available.
