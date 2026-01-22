# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to the LLM llama-3.2-1B-Instruct, then prints the
# what the LLM returns as text to stdout.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM llama-3.2-1B-Instruct from Hugging Face and wraps
# it for LangChain using HuggingFacePipeline.
# The code maintains chat history using the Message API with roles: system, human, ai.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of messages maintaining chat history (system, human, ai roles)
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - verbose: Boolean flag indicating if verbose tracing is enabled (set by get_user_input node)
    - skip_llm: Boolean flag indicating if LLM should be skipped (for verbose/quiet commands)
    - is_empty_input: Boolean flag indicating if user input is empty/whitespace (set by get_user_input node)

    State Flow:
    1. Initial state: messages list initialized with system message, other fields empty/default
    2. After get_user_input: user_input, should_exit, verbose, skip_llm, and is_empty_input are populated
    3. After get_user_input (valid input): HumanMessage added to messages
    4. After call_llama: AIMessage added to messages with LLM response
    5. After print_response: state unchanged (node only reads, doesn't write)

    The graph loops continuously with conditional routing from get_user_input:
        get_user_input -> [conditional] -> get_user_input (if empty input - loop back)
                              |
                              +-> END (if user wants to quit)
                              +-> call_llama (if valid input)
                              +-> print_response (if skip_llm, bypassing LLMs)
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    should_exit: bool
    verbose: bool
    skip_llm: bool
    is_empty_input: bool

def create_llama_llm():
    """
    Create and configure the Llama LLM using HuggingFace's transformers library.
    Downloads llama-3.2-1B-Instruct from HuggingFace Hub and wraps it
    for use with LangChain via HuggingFacePipeline.
    """
    # Get the optimal device for inference
    device = get_device()

    # Model identifier on HuggingFace Hub
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading Llama model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,      # Enable sampling for varied responses
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_p=0.95,          # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
        return_full_text=False,  # Return only the newly generated text, not the input prompt
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    print("Llama model loaded successfully!")
    return llm

def create_graph(llama_llm):
    """
    Create the LangGraph state graph with nodes for single model execution:
    1. get_user_input: Reads input from stdin and adds HumanMessage to chat history
    2. call_llama: Sends messages (with full chat history) to the Llama model
    3. print_response: Prints the model response to stdout
    4. print_response: Prints messages for verbose/quiet commands

    Graph structure with conditional routing:
        START -> get_user_input -> [conditional] -> call_llama -> print_response -+
                       ^                 |              |                          |
                       |                 |              +-> print_response (if skip_llm)
                       |                 +-> END (if user wants to quit)          |
                       |                                                           |
                       +-----------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    Users can type 'verbose' to enable tracing or 'quiet' to disable it.
    Chat history is maintained using the Message API with roles: system, human, ai.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    # This node reads a line of text from stdin and updates the state.
    # State changes:
    #   - user_input: Set to the text entered by the user
    #   - should_exit: Set to True if user typed quit/exit/q, False otherwise
    #   - verbose: Set to True if user typed "verbose", False if "quiet", otherwise unchanged
    #   - skip_llm: Set to True if user typed "verbose" or "quiet" (these are commands, not LLM inputs)
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.

        Reads state:
            - verbose: Current verbose mode setting (to preserve it)
        Updates state:
            - user_input: The raw text entered by the user
            - should_exit: True if user wants to quit, False otherwise
            - verbose: Updated if user typed "verbose" or "quiet"
            - skip_llm: True if user typed "verbose" or "quiet" (commands, not LLM inputs)
            - is_empty_input: True if user input is empty or only whitespace
        """
        verbose = state.get("verbose", False)
        
        # Print tracing info if verbose mode is enabled
        if verbose:
            print("\n[TRACE] Entering node: get_user_input")
            print(f"[TRACE] Current state - verbose: {verbose}, should_exit: {state.get('should_exit', False)}")
        
        # Display banner before each prompt
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit, 'verbose' for tracing, 'quiet' to disable tracing):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        # Check if input is empty or only whitespace
        is_empty = not user_input.strip()
        
        if is_empty:
            if verbose:
                print("[TRACE] Empty input detected - will loop back to get_user_input")
            print("Please enter some text (empty input ignored).")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": verbose,         # Preserve verbose setting
                "skip_llm": False,          # Not relevant for empty input
                "is_empty_input": True      # Signal that input is empty
            }

        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,        # Signal to exit the graph
                "verbose": verbose,         # Preserve verbose setting
                "skip_llm": False,          # Not relevant for exit
                "is_empty_input": False     # Not empty
            }

        # Check if user wants to toggle verbose mode
        if user_input.lower() == "verbose":
            if verbose:
                print("[TRACE] Verbose mode already enabled")
            else:
                print("Verbose tracing enabled")
            if verbose:
                print("[TRACE] Setting verbose=True, skip_llm=True")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True,            # Enable verbose mode
                "skip_llm": True,           # Skip LLM for this command
                "is_empty_input": False    # Not empty
            }
        
        if user_input.lower() == "quiet":
            if verbose:
                print("[TRACE] Disabling verbose mode")
            else:
                print("Verbose tracing disabled")
            if verbose:
                print("[TRACE] Setting verbose=False, skip_llm=True")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": False,           # Disable verbose mode
                "skip_llm": True,           # Skip LLM for this command
                "is_empty_input": False    # Not empty
            }

        # Any other input - continue to LLM
        if verbose:
            print(f"[TRACE] User input received: '{user_input}'")
            print("[TRACE] Setting skip_llm=False (will proceed to LLM)")
            print("[TRACE] Adding HumanMessage to chat history")
        
        # Add user input as HumanMessage to chat history
        human_message = HumanMessage(content=user_input)
        
        return {
            "messages": [human_message],  # Add HumanMessage to messages list
            "user_input": user_input,
            "should_exit": False,           # Signal to proceed to LLM
            "verbose": verbose,             # Preserve verbose setting
            "skip_llm": False,              # Proceed to LLM
            "is_empty_input": False        # Not empty
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    # This node takes the messages (with full chat history) from state, sends them
    # to the Llama model, and stores the response as an AIMessage in state.
    # State changes:
    #   - messages: Adds AIMessage with the Llama model's generated response
    def call_llama(state: AgentState) -> dict:
        """
        Node that invokes the Llama model with the full chat history.
        The model receives all previous messages (system, human, ai) for context.

        Reads state:
            - messages: The full chat history (system, human, ai messages)
            - verbose: Whether to print tracing information
        Updates state:
            - messages: Adds AIMessage with the text generated by the Llama model
        """
        verbose = state.get("verbose", False)
        # Get messages directly from state - should always exist due to add_messages reducer
        # Use direct access to ensure we get the actual messages list
        messages = state.get("messages", [])
        if not messages:
            # This shouldn't happen, but log it if it does
            if verbose:
                print("[TRACE] WARNING: messages list is empty!")
            messages = []

        # Print tracing info if verbose mode is enabled
        if verbose:
            print("\n[TRACE] Entering node: call_llama")
            print(f"[TRACE] Current chat history has {len(messages)} messages")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                print(f"[TRACE]   Message {i}: {msg_type} - {content_preview}")

        print("\nProcessing your input with Llama...")

        if verbose:
            print("[TRACE] Invoking Llama model with full chat history...")

        # Get the tokenizer from the pipeline to format messages correctly
        pipeline_obj = llama_llm.pipeline
        tokenizer = pipeline_obj.tokenizer
        
        # Convert LangChain messages to the format expected by the chat template
        # Format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        if verbose:
            print(f"[TRACE] Converted {len(formatted_messages)} messages to chat template format")
            for i, fm in enumerate(formatted_messages):
                content_preview = str(fm["content"])[:50] + "..." if len(str(fm["content"])) > 50 else str(fm["content"])
                print(f"[TRACE]   Formatted message {i}: role={fm['role']}, content={content_preview}")
        
        # Use the tokenizer's chat template to format the messages properly
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            # Format messages using the chat template
            formatted_prompt = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if verbose:
                print(f"[TRACE] Formatted prompt length: {len(formatted_prompt)} chars")
                print(f"[TRACE] Formatted prompt (first 500 chars):\n{formatted_prompt[:500]}...")
                print(f"[TRACE] Formatted prompt (last 200 chars):\n...{formatted_prompt[-200:]}")
            
            # Get the prompt with generation prompt to extract only new content later
            prompt_with_gen = formatted_prompt
            
            # Invoke with the formatted prompt string
            # Note: HuggingFacePipeline might return full text, so we'll extract only new part
            response = llama_llm.invoke(formatted_prompt)
        else:
            # Fallback: invoke with messages directly
            prompt_with_gen = None
            response = llama_llm.invoke(messages)

        if verbose:
            print(f"[TRACE] Llama response type: {type(response)}")
            print(f"[TRACE] Llama response received: '{response}'")
            print("[TRACE] Adding AIMessage to chat history")
            print("[TRACE] Exiting node: call_llama")

        # Extract the response content (could be a string, Message object, or list)
        if isinstance(response, BaseMessage):
            response_content = response.content
        elif isinstance(response, list) and len(response) > 0:
            # HuggingFacePipeline might return a list of dictionaries
            if isinstance(response[0], dict) and "generated_text" in response[0]:
                response_content = response[0]["generated_text"]
            elif isinstance(response[0], dict) and "text" in response[0]:
                response_content = response[0]["text"]
            else:
                response_content = str(response[0])
        else:
            response_content = str(response)
        
        # Clean up: if response includes the prompt, extract only the new part
        # This can happen if the pipeline returns the full generated text including the prompt
        response_str = response_content.strip()
        
        # If we used the chat template, extract only the newly generated part
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None and prompt_with_gen:
            # Remove the prompt from the beginning if it's included
            if response_str.startswith(prompt_with_gen):
                response_str = response_str[len(prompt_with_gen):].strip()
            
            # Remove special tokens that might be at the start/end
            # Llama chat template uses tokens like <|start_header_id|>assistant<|end_header_id|>
            special_tokens = [
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ]
            for token in special_tokens:
                # Remove from start
                while response_str.startswith(token):
                    response_str = response_str[len(token):].strip()
                # Remove from end
                while response_str.endswith(token):
                    response_str = response_str[:-len(token)].strip()
                # Remove any occurrences (in case they're in the middle)
                response_str = response_str.replace(token, "").strip()
        
        # Create AIMessage with the cleaned response
        ai_message = AIMessage(content=response_str)

        # Return the AIMessage to be added to the messages list
        return {"messages": [ai_message]}

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    # This node prints the latest AI response from the chat history.
    # State changes:
    #   - No changes (this node only reads state, doesn't modify it)
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the latest AI response from the chat history.
        For verbose/quiet commands, it just returns without printing.

        Reads state:
            - messages: The full chat history to extract the latest AI response
            - verbose: Whether to print tracing information
            - skip_llm: Whether LLM was skipped (for verbose/quiet commands)
        Updates state:
            - Nothing (returns empty dict, state unchanged)
        """
        verbose = state.get("verbose", False)
        skip_llm = state.get("skip_llm", False)
        messages = state.get("messages", [])

        # Print tracing info if verbose mode is enabled
        if verbose:
            print("\n[TRACE] Entering node: print_response")
            print(f"[TRACE] Current state - skip_llm: {skip_llm}, verbose: {verbose}")
            print(f"[TRACE] Chat history has {len(messages)} messages")

        # If LLM was skipped (verbose/quiet commands), don't print LLM response
        if skip_llm:
            if verbose:
                print("[TRACE] LLMs were skipped (verbose/quiet command), no response to print")
                print("[TRACE] Exiting node: print_response")
            return {}

        # Find the latest AI message in the chat history
        ai_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break

        if ai_response:
            print("\n" + "=" * 50)
            print("MODEL RESPONSE")
            print("=" * 50)
            print("\n" + "-" * 50)
            print("Llama Response:")
            print("-" * 50)
            print(ai_response)
            print("\n" + "=" * 50)
            
            if verbose:
                print("[TRACE] Response printed successfully")
        else:
            print("\nNo AI response available in chat history")
            if verbose:
                print("[TRACE] No AIMessage found in messages")

        if verbose:
            print("[TRACE] Exiting node: print_response")

        # Return empty dict - no state updates from this node
        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    # This function examines the state and determines which node to go to next.
    # It's used for conditional edges after get_user_input.
    # Three-way conditional routing:
    #   1. Empty input -> loop back to get_user_input
    #   2. User wants to quit -> END
    #   3. User entered valid input -> proceed to call_llm or print_response
    def route_after_input(state: AgentState):
        """
        Routing function that determines the next node based on state.
        Implements conditional branching from get_user_input.

        Examines state:
            - is_empty_input: If True, loop back to get_user_input
            - should_exit: If True, terminate the graph
            - skip_llm: If True, skip LLM and go directly to print_response

        Returns:
            - "get_user_input": If input is empty (loop back)
            - "__end__": If user wants to quit
            - "print_response": If user typed verbose/quiet (skip LLM)
            - "call_llama": Otherwise (default - proceed to LLM)
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n[TRACE] Routing function: route_after_input")
            print(f"[TRACE] is_empty_input: {state.get('is_empty_input', False)}")
            print(f"[TRACE] should_exit: {state.get('should_exit', False)}, skip_llm: {state.get('skip_llm', False)}")
        
        # First check: Empty input -> loop back to get_user_input
        if state.get("is_empty_input", False):
            if verbose:
                print("[TRACE] Routing to: get_user_input (empty input - loop back)")
            return "get_user_input"
        
        # Second check: User wants to exit
        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Routing to: END")
            return END

        # Third check: Should skip LLM (verbose/quiet commands)
        if state.get("skip_llm", False):
            if verbose:
                print("[TRACE] Routing to: print_response (skipping LLM)")
            return "print_response"
        
        # Default: Route to Llama for all valid inputs
        if verbose:
            print("[TRACE] Routing to: call_llama (default)")
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    # Create a StateGraph with our defined state structure
    graph_builder = StateGraph(AgentState)

    # Add all nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("print_response", print_response)

    # Define edges:
    # 1. START -> get_user_input (always start by getting user input)
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> get_user_input OR call_llama OR print_response OR END
    #    Uses route_after_input to decide based on state:
    #    - Empty input -> loop back to get_user_input
    #    - Quit command -> END
    #    - Verbose/quiet -> print_response (skip LLM)
    #    - Valid input -> call_llama
    graph_builder.add_conditional_edges(
        "get_user_input",      # Source node
        route_after_input,      # Routing function that examines state
        {
            "get_user_input": "get_user_input", # Empty input -> loop back to itself
            "call_llama": "call_llama",         # Valid input -> route to Llama
            "print_response": "print_response", # Verbose/quiet -> skip LLMs
            END: END                            # Quit command -> terminate graph
        }
    )

    # 3. call_llama -> print_response
    #    After getting LLM response, print it
    graph_builder.add_edge("call_llama", "print_response")

    # 4. print_response -> get_user_input (loop back for next input)
    #    This creates the continuous loop - after printing, go back to get more input
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def save_graph_mermaid(graph, filename="lg_graph.mmd"):
    """
    Save the graph as Mermaid diagram text format.
    This can be viewed online at https://mermaid.live/ or in many markdown viewers.
    """
    try:
        # Get the Mermaid text representation
        mermaid_text = graph.get_graph(xray=True).draw_mermaid()
        
        # Write to file
        with open(filename, "w") as f:
            f.write(mermaid_text)
        
        print(f"Mermaid diagram saved to {filename}")
        print(f"You can view it online at: https://mermaid.live/")
        print(f"Or copy the contents and paste into any Mermaid-compatible viewer")
        return mermaid_text
    except Exception as e:
        print(f"Could not save Mermaid diagram: {e}")
        return None

def print_graph_structure(graph):
    """
    Print the graph structure as text for quick inspection.
    """
    try:
        graph_structure = graph.get_graph(xray=True)
        print("\n" + "=" * 50)
        print("Graph Structure:")
        print("=" * 50)
        print(graph_structure.draw_ascii())
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"Could not print graph structure: {e}")

def visualize_graph(graph, save_png=True, save_mermaid=True, print_ascii=True):
    """
    Comprehensive graph visualization function.
    
    Args:
        graph: The compiled LangGraph
        save_png: Whether to save PNG image (requires grandalf)
        save_mermaid: Whether to save Mermaid text file
        print_ascii: Whether to print ASCII representation
    """
    print("\n" + "=" * 50)
    print("Graph Visualization")
    print("=" * 50)
    
    # Print ASCII representation
    if print_ascii:
        print_graph_structure(graph)
    
    # Save Mermaid text file (most reliable)
    if save_mermaid:
        mermaid_text = save_graph_mermaid(graph)
        if mermaid_text:
            print(f"\nMermaid code preview (first 500 chars):")
            print("-" * 50)
            print(mermaid_text[:500] + "..." if len(mermaid_text) > 500 else mermaid_text)
            print("-" * 50)
    
    # Save PNG image (requires grandalf)
    if save_png:
        save_graph_image(graph)
    
    print("=" * 50 + "\n")

def main():
    """
    Main function that orchestrates the simple agent workflow:
    1. Initialize the LLM
    2. Create the LangGraph
    3. Save the graph visualization
    4. Run the graph once (it loops internally until user quits)

    The graph handles all looping internally through its edge structure:
    - get_user_input: Prompts and reads from stdin
    - call_llm: Processes input through the LLM
    - print_response: Outputs the response, then loops back to get_user_input

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 50)
    print("LangGraph Agent with Llama Model (Chat History Enabled)")
    print("=" * 50)
    print()

    # Step 1: Create and configure the Llama LLM
    print("Loading model...")
    llama_llm = create_llama_llm()

    # Step 2: Build the LangGraph with the Llama LLM
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    # This happens BEFORE any graph execution, showing the graph structure
    print("\nGenerating graph visualization...")
    # Save with unique filenames for number5
    save_graph_image(graph, filename="lg_graph_number5.png")
    save_graph_mermaid(graph, filename="lg_graph_number5.mmd")
    print_graph_structure(graph)

    # Step 4: Run the graph - it will loop internally until user quits
    # Create initial state with system message and empty/default values
    # The graph will loop continuously, updating state as it goes:
    #   - get_user_input displays banner, populates user_input, should_exit, verbose, and skip_llm
    #   - get_user_input adds HumanMessage to messages for valid input
    #   - call_llama adds AIMessage to messages with LLM response
    #   - print_response displays output, then loops back to get_user_input
    system_message = SystemMessage(content="You are a helpful AI assistant. You maintain context from previous messages in the conversation.")
    initial_state: AgentState = {
        "messages": [system_message],  # Initialize with system message
        "user_input": "",
        "should_exit": False,
        "verbose": False,      # Start with verbose mode disabled
        "skip_llm": False,     # Start with normal flow
        "is_empty_input": False # Start with non-empty (will be set by get_user_input)
    }

    # Single invocation - the graph loops internally via print_response -> get_user_input
    # The graph only exits when route_after_input returns END (user typed quit/exit/q)
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()
