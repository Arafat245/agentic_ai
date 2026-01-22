"""
Enhanced Chat Agent - Understanding LLMs from the Inside

This enhanced chat agent demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input
4. Streaming responses (token-by-token generation)
5. Context management (handling long conversations)
6. Saving and loading conversation history
7. Error handling and recovery

No pre-defined chat libraries - built from scratch to understand how it works!
"""

import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

# Model selection - you can change this to your favorite model
# Popular options:
#   - "allenai/OLMo-2-0425-1B-Instruct" (OLMo 2 1B Instruct - open, transparent model)
#   - "allenai/OLMo-2-0425-1B" (OLMo 2 1B base model)
#   - "meta-llama/Llama-3.2-1B-Instruct" (small, fast)
#   - "meta-llama/Llama-3.2-3B-Instruct" (better quality)
#   - "mistralai/Mistral-7B-Instruct-v0.2" (excellent quality)
#   - "Qwen/Qwen2.5-1.5B-Instruct" (good alternative)
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Generation parameters
MAX_NEW_TOKENS = 512          # Maximum tokens in response
TEMPERATURE = 0.7             # Lower = more focused, higher = more creative
TOP_P = 0.9                   # Nucleus sampling parameter
DO_SAMPLE = True              # Use sampling (True) or greedy decoding (False)

# Context management - Sliding Window with Summarization
MAX_CONTEXT_TOKENS = 4096           # Maximum context length
RECENT_MESSAGES_TO_KEEP = 6         # Keep this many recent exchanges in full detail
SUMMARIZATION_THRESHOLD = 0.75      # Trigger summarization at 75% of max context
SUMMARY_TOKEN_BUDGET = 200         # Target tokens for summarized history

# Conversation history toggle
USE_HISTORY = True                  # Set to False to disable conversation history
                                  # When False, each turn only sees: system prompt + current user message
                                  # This allows comparison of performance with/without history
                                  # 
                                  # HOW TO COMPARE:
                                  # 1. Start a conversation with history ON (default)
                                  # 2. Have a multi-turn conversation (e.g., ask follow-up questions)
                                  # 3. Toggle history OFF with /history command
                                  # 4. Try the same type of follow-up questions
                                  # 5. Observe: With history, model remembers context. Without history, each turn is independent.
                                  #
                                  # Example multi-turn test:
                                  #   Turn 1: "My name is Alice and I like Python programming"
                                  #   Turn 2: "What's my name?" (with history: remembers Alice, without: doesn't know)
                                  #   Turn 3: "What programming language do I like?" (with history: remembers Python, without: doesn't know)

# File paths
HISTORY_DIR = "chat_history"  # Directory to save conversation history

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_history_dir():
    """Create history directory if it doesn't exist"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        print(f"✓ Created directory: {HISTORY_DIR}")

def save_conversation(chat_history: List[Dict], filename: Optional[str] = None):
    """Save conversation history to a JSON file"""
    ensure_history_dir()
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{HISTORY_DIR}/conversation_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, indent=2, ensure_ascii=False)
    print(f"✓ Conversation saved to: {filename}")
    return filename

def load_conversation(filename: str) -> List[Dict]:
    """Load conversation history from a JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_chat_for_model(chat_history: List[Dict], tokenizer, add_generation_prompt: bool = True):
    """
    Format chat history for the model using apply_chat_template.
    Handles cases where the template might not support certain parameters.
    """
    try:
        # Try with add_generation_prompt first
        return tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        )
    except (TypeError, AttributeError) as e:
        # Some models might not support add_generation_prompt
        try:
            return tokenizer.apply_chat_template(
                chat_history,
                return_tensors="pt"
            )
        except Exception as e2:
            # If apply_chat_template doesn't work, try manual formatting
            print(f"⚠ Warning: apply_chat_template failed, using fallback formatting")
            # Simple fallback: just encode the last user message
            if chat_history:
                last_user_msg = next((m for m in reversed(chat_history) if m["role"] == "user"), None)
                if last_user_msg:
                    return tokenizer.encode(last_user_msg["content"], return_tensors="pt")
            raise e2

def summarize_messages(messages: List[Dict], model, tokenizer, max_summary_tokens: int) -> str:
    """
    Summarize a list of messages into a concise summary.
    Uses the model itself to generate the summary.
    """
    if not messages:
        return ""
    
    # Filter out system messages (like existing summaries) from what we're summarizing
    messages_to_summarize = [msg for msg in messages if msg.get("role") != "system"]
    
    if not messages_to_summarize:
        return ""
    
    # Create a prompt for summarization
    conversation_text = ""
    for msg in messages_to_summarize:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Skip very long messages to avoid token limit issues
        if len(content) > 500:
            content = content[:500] + "..."
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n"
    
    # Truncate if conversation is too long
    conv_tokens = len(tokenizer.encode(conversation_text))
    if conv_tokens > 2000:  # Limit input to summarization
        # Take first and last parts
        lines = conversation_text.split("\n")
        mid = len(lines) // 2
        conversation_text = "\n".join(lines[:mid//2] + ["... (middle of conversation) ..."] + lines[-mid//2:])
    
    summary_prompt = f"""Summarize the following conversation concisely, preserving key information, facts, and context:

{conversation_text}

Summary:"""
    
    # Tokenize and generate summary
    try:
        input_ids = tokenizer.encode(summary_prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_summary_tokens,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused summaries
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        summary_tokens = outputs[0][input_ids.shape[1]:]
        summary = tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()
        
        # Clean up summary
        if summary.startswith("Summary:"):
            summary = summary[8:].strip()
        
        return summary if summary else f"Previous conversation with {len(messages_to_summarize)} messages."
    except Exception as e:
        print(f"⚠ Error generating summary: {e}")
        # Fallback: create a simple summary
        num_exchanges = len([m for m in messages_to_summarize if m.get("role") == "user"])
        return f"Previous conversation with {num_exchanges} exchanges."

def manage_context_sliding_window(
    chat_history: List[Dict], 
    tokenizer, 
    model,
    max_tokens: int,
    recent_count: int,
    threshold: float
) -> List[Dict]:
    """
    Advanced context management using sliding window with summarization.
    
    Strategy:
    1. Keep system prompt (always)
    2. Keep recent messages in full detail (last N exchanges)
    3. When context exceeds threshold, summarize older messages
    4. Maintain: [system, summary, recent_messages]
    
    This approach preserves important early context while managing token budget.
    """
    if not chat_history:
        return chat_history
    
    # Separate system prompt from messages
    system_msg = chat_history[0] if chat_history and chat_history[0].get("role") == "system" else None
    messages = chat_history[1:] if system_msg else chat_history
    
    if not messages:
        return chat_history
    
    # Check current token count
    try:
        test_history = [system_msg] + messages if system_msg else messages
        test_input = format_chat_for_model(test_history, tokenizer, add_generation_prompt=True)
        current_tokens = test_input.shape[1]
    except Exception as e:
        print(f"⚠ Could not check token count: {e}")
        return chat_history
    
    threshold_tokens = int(max_tokens * threshold)
    
    # If we're under threshold, no action needed
    if current_tokens <= threshold_tokens:
        return chat_history
    
    print(f"⚠ Context management triggered: {current_tokens} tokens (threshold: {threshold_tokens})")
    
    # Find where to split: keep recent messages, summarize older ones
    # Count exchanges (user+assistant pairs)
    num_exchanges = 0
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            num_exchanges += 1
        i += 1
    
    # Determine how many recent exchanges to keep
    exchanges_to_keep = min(recent_count, num_exchanges)
    
    # Split messages: recent (keep) vs old (summarize)
    if exchanges_to_keep >= num_exchanges:
        # All messages are recent, just truncate oldest if needed
        return chat_history
    
    # Find the split point (keep last N exchanges)
    # Also check for existing summary messages
    messages_to_keep = []
    messages_to_summarize = []
    existing_summary_msg = None
    
    # Check if there's already a summary in the messages (look for system messages with summary content)
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and "[Previous conversation summary:" in msg.get("content", ""):
            existing_summary_msg = msg
            # Everything before this summary should be included in what we summarize
            messages_to_summarize = messages[:i]
            messages = messages[i+1:]  # Continue with messages after summary
            break
    
    # Count backwards to find where to split recent messages
    exchange_count = 0
    split_index = len(messages)
    
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            exchange_count += 1
            if exchange_count > exchanges_to_keep:
                split_index = i + 1
                break
    
    messages_to_keep = messages[split_index:]
    # Add messages before split to what we'll summarize
    messages_to_summarize.extend(messages[:split_index])
    
    # Build new history: system + summary + recent messages
    new_history = []
    if system_msg:
        new_history.append(system_msg)
    
    # Summarize old messages if we have any
    if messages_to_summarize:
        print(f"📝 Summarizing {len(messages_to_summarize)} old messages...")
        
        # If we have an existing summary, include it in the summarization context
        if existing_summary_msg:
            # Extract old summary text
            old_summary = existing_summary_msg.get("content", "")
            if "[Previous conversation summary:" in old_summary:
                old_summary = old_summary.replace("[Previous conversation summary:", "").replace("]", "").strip()
            # Add old summary as context
            messages_to_summarize.insert(0, {
                "role": "system",
                "content": f"Previous summary: {old_summary}"
            })
        
        summary_text = summarize_messages(messages_to_summarize, model, tokenizer, SUMMARY_TOKEN_BUDGET)
        
        # Add summary as a special message
        summary_msg = {
            "role": "system",  # Use system role for summary to distinguish it
            "content": f"[Previous conversation summary: {summary_text}]"
        }
        new_history.append(summary_msg)
    
    # Add recent messages
    new_history.extend(messages_to_keep)
    
    # Verify the new history fits
    try:
        test_input = format_chat_for_model(new_history, tokenizer, add_generation_prompt=True)
        final_tokens = test_input.shape[1]
        print(f"✓ Context managed: {final_tokens} tokens (kept {len(messages_to_keep)} recent messages, summarized {len(messages_to_summarize)} old messages)")
    except Exception as e:
        print(f"⚠ Warning: Could not verify token count after summarization: {e}")
    
    return new_history

def generate_streaming(model, tokenizer, input_ids, attention_mask, generation_config):
    """
    Generate response token-by-token (streaming).
    This shows how tokens are generated one at a time using the model's generate
    method with a custom callback to stream tokens as they're generated.
    """
    generated_tokens = []
    
    # Use a simple approach: generate with a callback to stream tokens
    # We'll use the model's generate but decode incrementally
    with torch.no_grad():
        # Start generation - ensure input_ids is 2D: (batch_size, seq_len)
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)  # Remove extra batch dimension if present
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(0)
        
        generated_sequence = input_ids.clone()
        
        for step in range(generation_config["max_new_tokens"]):
            # Forward pass to get logits
            # Ensure inputs are on the same device as the model
            generated_sequence = generated_sequence.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            outputs = model(input_ids=generated_sequence, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Get logits for last token - shape: (batch_size, vocab_size)
            
            # Apply temperature
            if generation_config["do_sample"]:
                logits = logits / generation_config["temperature"]
                
                # Top-p (nucleus) sampling
                if generation_config["top_p"] < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > generation_config["top_p"]
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            
            # Ensure next_token is 2D: (batch_size, 1)
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(1)
            elif next_token.dim() == 0:
                next_token = next_token.unsqueeze(0).unsqueeze(1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Decode and print token (skip special tokens for display)
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(token_text, end="", flush=True)
            
            # Add to generated tokens
            token_id = next_token.item() if next_token.numel() == 1 else next_token[0, 0].item()
            generated_tokens.append(token_id)
            
            # Update sequence for next iteration
            # Ensure next_token is on the same device and has the right shape
            next_token = next_token.to(generated_sequence.device)
            # next_token should be shape (batch_size, 1)
            if next_token.dim() != 2:
                next_token = next_token.view(-1, 1)
            
            # Concatenate along sequence dimension
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)
    
    return generated_tokens

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this may take 1-2 minutes)...")
print(f"Model: {MODEL_NAME}\n")

try:
    # Load tokenizer (converts text to numbers and vice versa)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in half precision (float16) for efficiency
    # Use float16 on GPU, or float32 on CPU if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    model.eval()  # Set to evaluation mode (no training)
    
    print(f"✓ Model loaded successfully!")
    print(f"✓ Device: {model.device}")
    print(f"✓ Precision: {dtype}")
    print(f"✓ Context length: {MAX_CONTEXT_TOKENS} tokens\n")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    print("\nFull error traceback:")
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Make sure you have Hugging Face authentication set up")
    print("2. Check your internet connection")
    print("3. Verify the model name is correct")
    print("4. Ensure you have enough disk space and RAM")
    print("5. Make sure transformers version is >= 4.48 (required for OLMo 2)")
    exit(1)

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# CHAT LOOP
# ============================================================================

def print_help():
    """Print available commands"""
    print("\n" + "="*70)
    print("COMMANDS:")
    print("  /help, /h          - Show this help message")
    print("  /save, /s          - Save conversation to file")
    print("  /load <file>       - Load conversation from file")
    print("  /clear, /c         - Clear conversation history (keeps system prompt)")
    print("  /stats             - Show conversation statistics")
    print("  /stream            - Toggle streaming mode (on/off)")
    print("  /history, /hist     - Toggle conversation history (on/off)")
    print("                       When OFF: each turn only sees current message")
    print("                       When ON: maintains full conversation context")
    print("  /quit, /exit, /q   - Exit the chat")
    print("="*70 + "\n")

def print_stats(chat_history, tokenizer, use_history=True):
    """Print conversation statistics"""
    if not chat_history:
        print("No conversation yet.")
        return
    
    num_messages = len([m for m in chat_history if m["role"] != "system"])
    num_exchanges = num_messages // 2
    
    # Estimate token count based on history mode
    try:
        if use_history:
            test_input = format_chat_for_model(chat_history, tokenizer, add_generation_prompt=True)
        else:
            # When history is off, only count system + last message
            system_msg = chat_history[0] if chat_history and chat_history[0].get("role") == "system" else None
            last_msg = chat_history[-1] if chat_history else None
            if system_msg and last_msg:
                test_history = [system_msg, last_msg]
            elif last_msg:
                test_history = [last_msg]
            else:
                test_history = chat_history
            test_input = format_chat_for_model(test_history, tokenizer, add_generation_prompt=True)
        token_count = test_input.shape[1]
    except:
        token_count = "unknown"
    
    print(f"\n📊 Conversation Statistics:")
    print(f"   History mode: {'ON (full context)' if use_history else 'OFF (stateless)'}")
    print(f"   Total messages: {num_messages} ({num_exchanges} exchanges)")
    print(f"   Estimated tokens per turn: {token_count}")
    print(f"   Context limit: {MAX_CONTEXT_TOKENS} tokens")
    if not use_history:
        print(f"   Note: Each turn only uses system prompt + current message")
    print()

# Runtime toggles (can be changed during conversation)
USE_STREAMING = True
USE_HISTORY_RUNTIME = USE_HISTORY  # Runtime version that can be toggled

print("="*70)
print("Enhanced Chat Agent - Understanding LLMs from the Inside")
print("="*70)
print(f"\nChat started! Type '/help' for commands or '/quit' to exit.")
print(f"Conversation history: {'ON' if USE_HISTORY_RUNTIME else 'OFF'}")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye!")
        break
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q', '/quit', '/exit', '/q']:
        print("\nGoodbye!")
        # Offer to save before exiting
        save_choice = input("Save conversation? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_conversation(chat_history)
        break
    
    # Handle commands
    if user_input.startswith('/'):
        cmd = user_input.lower().split()[0]
        
        if cmd in ['/help', '/h']:
            print_help()
            continue
        elif cmd in ['/save', '/s']:
            save_conversation(chat_history)
            continue
        elif cmd == '/load':
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /load <filename>")
                continue
            try:
                chat_history = load_conversation(parts[1])
                print(f"✓ Loaded conversation from {parts[1]}")
                print(f"  {len(chat_history)} messages loaded")
            except Exception as e:
                print(f"❌ Error loading: {e}")
            continue
        elif cmd in ['/clear', '/c']:
            # Keep system prompt
            system_msg = chat_history[0] if chat_history and chat_history[0]["role"] == "system" else None
            chat_history = [system_msg] if system_msg else []
            print("✓ Conversation cleared")
            continue
        elif cmd == '/stats':
            print_stats(chat_history, tokenizer, USE_HISTORY_RUNTIME)
            continue
        elif cmd == '/stream':
            USE_STREAMING = not USE_STREAMING
            print(f"✓ Streaming mode: {'ON' if USE_STREAMING else 'OFF'}")
            continue
        elif cmd in ['/history', '/hist']:
            USE_HISTORY_RUNTIME = not USE_HISTORY_RUNTIME
            status = "ON" if USE_HISTORY_RUNTIME else "OFF"
            print(f"✓ Conversation history: {status}")
            if not USE_HISTORY_RUNTIME:
                print("  Note: Each turn will only see the current message (no context from previous turns)")
            else:
                print("  Note: Full conversation history will be maintained")
            continue
        else:
            print(f"Unknown command: {cmd}. Type '/help' for available commands.")
            continue
    
    # Skip empty inputs
    if not user_input:
        continue
    
    # ========================================================================
    # STEP 2: Handle conversation history based on USE_HISTORY_RUNTIME flag
    # ========================================================================
    if USE_HISTORY_RUNTIME:
        # Add user message to chat history (maintains full context)
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # ========================================================================
        # STEP 3: Context Management - Sliding Window with Summarization
        # ========================================================================
        chat_history = manage_context_sliding_window(
            chat_history, 
            tokenizer, 
            model,
            MAX_CONTEXT_TOKENS,
            RECENT_MESSAGES_TO_KEEP,
            SUMMARIZATION_THRESHOLD
        )
        
        # Use full chat history for tokenization
        history_for_model = chat_history
    else:
        # History is OFF: only use system prompt + current user message
        # Still maintain chat_history for display/stats, but don't use it for generation
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Create minimal context: only system prompt + current user message
        system_msg = chat_history[0] if chat_history and chat_history[0].get("role") == "system" else None
        if system_msg:
            history_for_model = [
                system_msg,
                {"role": "user", "content": user_input}
            ]
        else:
            history_for_model = [
                {"role": "user", "content": user_input}
            ]
    
    # ========================================================================
    # STEP 4: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    try:
        input_ids = format_chat_for_model(history_for_model, tokenizer, add_generation_prompt=True).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Show context info when history is off
        if not USE_HISTORY_RUNTIME:
            token_count = input_ids.shape[1]
            print(f"[History OFF - Using {token_count} tokens (system + current message only)]")
    except Exception as e:
        print(f"❌ Error tokenizing: {e}")
        import traceback
        traceback.print_exc()
        # Remove the last user message and continue
        if chat_history and chat_history[-1].get("role") == "user":
            chat_history.pop()
        continue
    
    # ========================================================================
    # STEP 5: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    print("Assistant: ", end="", flush=True)
    
    try:
        if USE_STREAMING:
            # Streaming generation (token-by-token)
            generation_config = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P
            }
            generated_tokens = generate_streaming(
                model, tokenizer, input_ids, attention_mask, generation_config
            )
            assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print()  # New line after streaming
        else:
            # Non-streaming generation (all at once)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the newly generated tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(assistant_response)
    
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        # Remove the last user message and continue
        if chat_history and chat_history[-1].get("role") == "user":
            chat_history.pop()
        continue
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # Add the assistant's response to history
    # If USE_HISTORY_RUNTIME is True, this will be used in future turns
    # If False, it's only kept for display/stats purposes
    
    chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    # Show reminder when history is off
    if not USE_HISTORY_RUNTIME:
        print(f"[Note: History is OFF - this response won't be used as context in future turns]")
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT AGENTS WORK
# ============================================================================
"""
KEY CONCEPTS:

1. PLAIN TEXT vs TOKENIZED:

   PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

   TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

2. PROCESS EACH TURN:
   User input (text)
   ↓
   Add to chat_history (text)
   ↓
   Context management (truncate if too long)
   ↓
   Tokenize entire chat_history (text → numbers)
   ↓
   Model generates response (numbers, token-by-token if streaming)
   ↓
   Decode response (numbers → text)
   ↓
   Add response to chat_history (text)
   ↓
   Loop back to start

3. WHY FEED ENTIRE HISTORY?
   - The model has no memory between calls
   - Each generation is independent
   - To "remember" previous turns, we must include them in the input
   - This is why context length matters - longer conversations = more tokens

4. CONTEXT MANAGEMENT (Sliding Window with Summarization):
   - As conversation grows, token count increases
   - When context exceeds 75% of max, we trigger summarization
   - Our manage_context_sliding_window() function:
     * Keeps system prompt (always)
     * Keeps recent RECENT_MESSAGES_TO_KEEP exchanges in full detail
     * Summarizes older messages using the model itself
     * Maintains structure: [system, summary, recent_messages]
   - Benefits:
     * Preserves important early context (via summary)
     * Keeps recent context in full detail
     * Manages token budget efficiently
     * Can handle arbitrarily long conversations
   - Advanced alternatives include:
     * Vector databases for semantic search
     * Hierarchical memory architectures
     * Knowledge graphs for structured memory

5. STREAMING GENERATION:
   - Instead of generating all tokens at once, we generate token-by-token
   - This allows real-time display as the model "thinks"
   - Shows how autoregressive generation works
   - Each token depends on all previous tokens

6. ENHANCED FEATURES IN THIS AGENT:
   - Streaming responses (see tokens as they're generated)
   - Advanced context management (sliding window with summarization)
   - Save/load conversations (persistence)
   - Error handling (graceful recovery)
   - Commands for managing conversation
   - Statistics and monitoring

7. WHAT YOU'VE LEARNED:
   - How chat history is maintained (plain text)
   - How it's converted to tokens (tokenization)
   - How models generate responses (autoregressive generation)
   - How responses are decoded back to text
   - How context management works
   - How streaming generation works
   - The complete flow from user input to assistant response

This agent is built from scratch to show you exactly how everything works!
No magic, no hidden abstractions - just the fundamentals of LLM chat agents.
"""
