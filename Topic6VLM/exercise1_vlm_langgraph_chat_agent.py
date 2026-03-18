"""
Vision-Language LangGraph Chat Agent

A multi-turn chat agent that can hold a conversation about an uploaded image.
Uses LangGraph for structure and context management, Ollama + LLaVA for vision-language inference.

Features:
- Multi-turn conversation with image context
- LangGraph state management with message history
- Automatic history trimming to prevent context overflow
- Image resolution reduction option for faster inference
- Gradio interface for a polished UX
"""

import base64
import io
import sys
from pathlib import Path
from typing import TypedDict, Optional, Tuple

import ollama
from PIL import Image
from langgraph.graph import StateGraph, END

# Optional: Gradio for polished UI (install with: pip install gradio)
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "llava:latest"  # Vision model; run: ollama pull llava
MAX_HISTORY_MESSAGES = 50  # Trim to prevent context overflow
MAX_IMAGE_DIMENSION = 768   # Reduce resolution for faster inference (set to None to disable)
VLM_SYSTEM_PROMPT = (
    "You are a helpful vision assistant. You can see and understand images. "
    "When the user provides an image, describe what you see and answer their questions about it. "
    "Be specific and detailed when describing image content."
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class VLMAgentState(TypedDict):
    """
    State schema for the Vision-Language chat agent.

    Attributes:
        messages: Conversation history in Ollama format (list of {role, content, images?})
        current_image_b64: Base64-encoded image for the current conversation
        new_user_message: Incoming user message for this turn
        new_user_image: Optional new image for this turn (base64 or path)
        response: VLM response for this turn
        command: Special command: "exit", "verbose", "quiet", or None
        verbose: Whether to print debug traces
    """
    messages: list
    current_image_b64: Optional[str]
    new_user_message: str
    new_user_image: Optional[str]
    response: str
    command: Optional[str]
    verbose: bool


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def load_image_as_base64(
    source: str,
    max_dimension: Optional[int] = MAX_IMAGE_DIMENSION,
) -> str:
    """
    Load an image from file path or base64 string and return as base64.

    Optionally resizes the image to reduce resolution for faster inference.

    Args:
        source: File path or base64-encoded image string
        max_dimension: Max width/height; image is resized if larger. None to skip.

    Returns:
        Base64-encoded JPEG string
    """
    if source is None or not str(source).strip():
        raise ValueError("No image source provided")

    source = str(source).strip()

    # Handle Gradio file URL - extract local path (e.g. http://.../file=/tmp/gradio/xxx/image.png)
    if source.startswith("http") and "file=" in source:
        path_str = source.split("file=", 1)[-1].split("?")[0]
        source = path_str

    # Resolve to PIL Image: handle base64, data: URLs, or file path
    img = None
    if source.startswith("data:"):
        # data:image/png;base64,... format
        b64 = source.split("base64,", 1)[-1] if "base64," in source else source
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        # Try base64 first if it doesn't look like a valid file path
        path = Path(source)
        if not path.exists() and len(source) > 50:
            try:
                img_data = base64.b64decode(source)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception:
                img = None
        if img is None:
            # File path
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {source}")
            img = Image.open(path).convert("RGB")

    # Optionally resize to speed up inference
    if max_dimension and (img.width > max_dimension or img.height > max_dimension):
        ratio = min(max_dimension / img.width, max_dimension / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def add_user_message_node(state: VLMAgentState) -> dict:
    """
    Add the new user message (and optionally image) to the conversation history.

    Updates messages and current_image_b64 in state.
    """
    verbose = state.get("verbose", False)
    new_msg = state.get("new_user_message", "").strip()
    new_img = state.get("new_user_image")
    messages = list(state.get("messages", []))
    current_img = state.get("current_image_b64")

    if verbose:
        print("[TRACE] add_user_message_node: formatting new message")

    # If user provided a new image, load it
    if new_img:
        try:
            current_img = load_image_as_base64(new_img)
            if verbose:
                print("[TRACE] Loaded new image")
        except Exception as e:
            if verbose:
                print(f"[TRACE] Image load error: {e}")
            # Fallback: new_img may already be base64 from submit's immediate load
            if isinstance(new_img, str) and len(new_img) > 100:
                try:
                    base64.b64decode(new_img)
                    current_img = new_img
                    if verbose:
                        print("[TRACE] Using new_img as raw base64")
                except Exception:
                    pass
            # Else keep previous image if load fails

    # Prepend system prompt on first message (helps vision models respond correctly)
    if not messages:
        messages.append({"role": "system", "content": VLM_SYSTEM_PROMPT})

    # Build Ollama-format message
    msg_dict: dict = {"role": "user", "content": new_msg}
    if current_img:
        msg_dict["images"] = [current_img]
    elif new_img and verbose:
        print("[TRACE] WARNING: No image in message despite new_img provided")

    messages.append(msg_dict)

    return {
        "messages": messages,
        "current_image_b64": current_img,
    }


def call_vlm_node(state: VLMAgentState) -> dict:
    """
    Invoke the LLaVA model via Ollama with the full conversation history.
    """
    verbose = state.get("verbose", False)
    messages = state.get("messages", [])

    if verbose:
        print(f"[TRACE] call_vlm_node: invoking {MODEL_NAME} with {len(messages)} messages")

    try:
        has_img = any(m.get("images") for m in messages if isinstance(m, dict))
        print(f"[VLM] Sending {len(messages)} messages, image attached: {has_img}")
        if verbose:
            print(f"[TRACE] call_vlm_node: invoking {MODEL_NAME}")
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        content = response.get("message", {}).get("content", "")
    except Exception as e:
        content = f"[Error calling VLM: {e}]"
        if verbose:
            print(f"[TRACE] VLM error: {e}")

    # Append assistant response to history for next turn
    messages = messages + [{"role": "assistant", "content": content}]

    return {"messages": messages, "response": content}


def trim_history_node(state: VLMAgentState) -> dict:
    """
    Trim conversation history to prevent context overflow.
    Keeps the system message (if present) and the most recent messages.
    """
    messages = state.get("messages", [])
    verbose = state.get("verbose", False)

    if len(messages) <= MAX_HISTORY_MESSAGES:
        return {}

    # Preserve system message if it's first
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    rest = messages[1:] if system_msg else messages
    # Keep system + last N-1 messages
    trimmed = ([system_msg] + rest[-(MAX_HISTORY_MESSAGES - 1):]) if system_msg else rest[-MAX_HISTORY_MESSAGES:]
    if verbose:
        print(f"[TRACE] trim_history: reduced from {len(messages)} to {len(trimmed)} messages")

    return {"messages": trimmed}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_vlm_graph():
    """
    Build the LangGraph for the Vision-Language chat agent.

    Flow:
        add_user_message -> call_vlm -> trim_history -> END

    Each invocation processes one user turn. The caller maintains state
    across invocations for multi-turn conversation.
    """
    workflow = StateGraph(VLMAgentState)

    workflow.add_node("add_user_message", add_user_message_node)
    workflow.add_node("call_vlm", call_vlm_node)
    workflow.add_node("trim_history", trim_history_node)

    workflow.set_entry_point("add_user_message")
    workflow.add_edge("add_user_message", "call_vlm")
    workflow.add_edge("call_vlm", "trim_history")
    workflow.add_edge("trim_history", END)

    return workflow.compile()


# =============================================================================
# CONVERSATION RUNNER
# =============================================================================

class VLMConversation:
    """
    Manages multi-turn conversation state and invokes the LangGraph.
    """

    def __init__(self, verbose: bool = False):
        self.graph = create_vlm_graph()
        self.state: VLMAgentState = {
            "messages": [],
            "current_image_b64": None,
            "new_user_message": "",
            "new_user_image": None,
            "response": "",
            "command": None,
            "verbose": verbose,
        }
        self.last_image_path: Optional[str] = None  # Track to detect new uploads

    def clear_state(self) -> None:
        """Clear conversation state (e.g. when user uploads a new image)."""
        self.state["messages"] = []
        self.state["current_image_b64"] = None
        self.last_image_path = None

    def chat(self, user_message: str, image_source: Optional[str] = None) -> str:
        """
        Process one user turn and return the assistant response.

        Args:
            user_message: User's text message
            image_source: Optional image (file path or base64). If provided, updates
                         the conversation image. If None, uses the existing image.

        Returns:
            Assistant's response text
        """
        self.state["new_user_message"] = user_message
        self.state["new_user_image"] = image_source

        result = self.graph.invoke(self.state)

        # Update state for next turn
        self.state["messages"] = result.get("messages", self.state["messages"])
        self.state["current_image_b64"] = result.get(
            "current_image_b64", self.state["current_image_b64"]
        )
        self.state["response"] = result.get("response", "")

        return self.state["response"]

    def set_image(self, image_source: str) -> None:
        """Set the current conversation image without sending a message."""
        try:
            b64 = load_image_as_base64(image_source)
            self.state["current_image_b64"] = b64
        except Exception as e:
            print(f"[Warning] Could not load image: {e}")


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_interface():
    """Create and return a Gradio Blocks interface for the VLM chat agent."""
    conv = VLMConversation(verbose=False)

    with gr.Blocks(title="VLM LangGraph Chat Agent") as demo:
        gr.Markdown("# Vision-Language LangGraph Chat Agent")
        gr.Markdown(
            "Upload an image and ask questions about it. The agent uses LLaVA via Ollama "
            "and LangGraph for multi-turn conversation."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    height=300,
                )
                gr.Markdown(
                    "Upload an image first, then type your question in the chat below. "
                    "You can upload a new image anytime to switch context."
                )
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=400)
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask a question about the image...",
                    lines=2,
                )
                submit_btn = gr.Button("Send")

        def _resolve_image_source(image, message: str) -> Tuple[Optional[str], str]:
            """Resolve image from Gradio input or from a pasted Gradio file URL.
            Returns (img_source, effective_message). With type='filepath', Gradio passes a path string."""
            if image is not None:
                if isinstance(image, str):
                    return image, message
                if isinstance(image, dict) and "path" in image:
                    return image["path"], message
                if isinstance(image, dict) and "url" in image:
                    return image["url"], message
                if hasattr(image, "shape"):
                    pil_img = Image.fromarray(image)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    return base64.b64encode(buffer.getvalue()).decode("utf-8"), message
                return str(image), message
            # Fallback: message might contain a Gradio file URL (user pasted it)
            msg = message.strip()
            if msg.startswith("http") and "file=" in msg:
                path_str = msg.split("file=", 1)[-1].split("?")[0].strip()
                if path_str and Path(path_str).exists():
                    return path_str, "Describe this image."
            return None, message

        def submit(message, history, image):
            if not message.strip():
                return history, ""
            img_source, effective_msg = _resolve_image_source(image, message)
            if img_source is None:
                history = list(history) if history else []
                history.append({"role": "user", "content": message})
                history.append({
                    "role": "assistant",
                    "content": "Please upload an image first using the panel on the left, then ask your question.",
                })
                return history, ""
            # 1. Read image into memory immediately (avoid temp file cleanup issues)
            try:
                img_base64 = load_image_as_base64(img_source)
            except Exception as e:
                history = list(history) if history else []
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"Error loading image: {e}"})
                return history, ""
            # 2. Clear conversation when user uploads a new image (different from previous)
            current_path = img_source if isinstance(img_source, str) and Path(img_source).exists() else None
            if current_path and current_path != conv.last_image_path:
                conv.clear_state()
            conv.last_image_path = current_path
            try:
                response = conv.chat(effective_msg, img_base64)
            except Exception as e:
                response = f"Error: {e}"
            # Gradio 6.0 uses messages format: list of {role, content} dicts
            history = list(history) if history else []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""  # Clear message only; image stays for follow-ups

        submit_btn.click(
            fn=submit,
            inputs=[msg_input, chatbot, image_input],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=submit,
            inputs=[msg_input, chatbot, image_input],
            outputs=[chatbot, msg_input],
        )

    return demo


def _launch_demo():
    """Launch the Gradio demo with theme in launch() for Gradio 6.x."""
    demo = create_gradio_interface()
    demo.launch(theme=gr.themes.Soft())


# =============================================================================
# CLI INTERFACE
# =============================================================================

def run_cli():
    """
    Run the agent from the command line.
    User provides image path first, then multi-turn chat.
    """
    print("=" * 60)
    print("Vision-Language LangGraph Chat Agent (CLI)")
    print("=" * 60)
    print("\nCommands: quit/exit to exit, verbose/quiet for tracing")
    print("First, enter the path to an image file.\n")

    conv = VLMConversation(verbose=False)
    image_path = None

    while True:
        if image_path is None:
            prompt = "Enter image path: "
        else:
            prompt = "You: "

        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "verbose":
            conv.state["verbose"] = True
            print("[System] Verbose mode enabled")
            continue

        if user_input.lower() == "quiet":
            conv.state["verbose"] = False
            print("[System] Verbose mode disabled")
            continue

        if image_path is None:
            # First input is image path
            path = Path(user_input)
            if not path.exists():
                print(f"[Error] File not found: {user_input}")
                continue
            image_path = user_input
            print("Image loaded. Now ask questions about it (or type 'new' to load a different image).")
            continue

        if user_input.lower() == "new":
            image_path = None
            print("Enter a new image path:")
            continue

        print("\nProcessing...")
        try:
            response = conv.chat(user_input, image_path)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"[Error] {e}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point: launch Gradio if available, else CLI."""
    if HAS_GRADIO and "--cli" not in sys.argv:
        print("Launching Gradio interface...")
        _launch_demo()
    else:
        run_cli()


if __name__ == "__main__":
    main()
