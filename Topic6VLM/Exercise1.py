import base64
import io
from typing import List, TypedDict, Optional

from PIL import Image

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# -------------------------
# Utilities
# -------------------------

def resize_image(image_path: str, max_size: int = 768) -> str:
    """
    Resize image to max_size on longest side and return base64 string.
    """
    image = Image.open(image_path)

    width, height = image.size
    scale = min(max_size / max(width, height), 1.0)
    new_size = (int(width * scale), int(height * scale))

    image = image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded


# -------------------------
# LangGraph State
# -------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    image_base64: Optional[str]


# -------------------------
# Model
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)


# -------------------------
# Nodes
# -------------------------

def vision_chat_node(state: AgentState):
    """
    Main reasoning node.
    Injects image only on first turn.
    """

    messages = state["messages"]
    image_base64 = state.get("image_base64")

    if image_base64:
        # Attach image only once
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": messages[-1].content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ]
        )
        new_messages = messages[:-1] + [image_message]
        state["image_base64"] = None  # Clear after first use
    else:
        new_messages = messages

    response = llm.invoke(new_messages)

    return {"messages": new_messages + [response]}


# -------------------------
# Graph Construction
# -------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("vision_chat", vision_chat_node)

graph_builder.set_entry_point("vision_chat")
graph_builder.add_edge("vision_chat", END)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)


# -------------------------
# Interactive Chat
# -------------------------

def chat_with_image(image_path: str):
    image_base64 = resize_image(image_path)

    thread_id = "vision-thread"

    print("\nImage loaded. You can now ask questions.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        state = {
            "messages": [HumanMessage(content=user_input)],
            "image_base64": image_base64,
        }

        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        ai_message = result["messages"][-1]
        print("\nAI:", ai_message.content, "\n")

        image_base64 = None  # only used first turn


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    chat_with_image("your_image.jpg")
