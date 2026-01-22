# Running an LLM: A Comprehensive Guide to Large Language Models

This repository contains implementations and experiments for understanding Large Language Models (LLMs) from the ground up, including model evaluation, chat agent development, and performance analysis.

## Table of Contents

- [Learning Goals](#learning-goals)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Tasks and Implementations](#tasks-and-implementations)
- [Results and Analysis](#results-and-analysis)
- [Key Findings](#key-findings)

---

## Learning Goals

### 1. Understanding LLMs at a High Level

**What is an LLM?**
- Large Language Models are neural networks trained on vast amounts of text data to predict the next token in a sequence
- They use transformer architecture with attention mechanisms to understand context
- Models are trained to generate human-like text by learning patterns from training data

**The Role of a Tokenizer:**
- **Tokenization**: Converts human-readable text into numerical tokens (IDs) that the model can process
- **Detokenization**: Converts model outputs (token IDs) back into human-readable text
- **Vocabulary**: A fixed set of tokens the model knows (typically 30K-100K+ tokens)
- **Special Tokens**: Control tokens like `<|start|>`, `<|end|>`, `<|user|>`, `<|assistant|>` that structure conversations

**Key Concepts:**
- **Context Window**: Maximum number of tokens a model can process in one go (e.g., 4K, 8K, 128K tokens)
- **Autoregressive Generation**: Model generates tokens one at a time, each depending on all previous tokens
- **Temperature**: Controls randomness in generation (lower = more focused, higher = more creative)
- **Top-p (Nucleus Sampling)**: Samples from tokens whose cumulative probability exceeds p

### 2. Range of LLMs and Benchmark Datasets

**Models Evaluated:**
- **Tiny/Small Models (1B-3B parameters)**:
  - `meta-llama/Llama-3.2-1B-Instruct` - Meta's efficient 1B model
  - `allenai/OLMo-2-0425-1B-Instruct` - Open, transparent model from AllenAI
  - `Qwen/Qwen2.5-0.5B-Instruct` - Alibaba's compact model
  - `Qwen/Qwen2.5-1.5B-Instruct` - Slightly larger Qwen variant
  - `microsoft/phi-2` - Microsoft's efficient 2.7B model

- **Medium Models (3B-7B parameters)**:
  - `Qwen/Qwen2.5-3B-Instruct` - Mid-size Qwen model
  - `Qwen/Qwen2.5-7B-Instruct` - Larger Qwen variant

**Major LLM Benchmark Datasets on Hugging Face:**
- **MMLU (Massive Multitask Language Understanding)**: 57 subjects across STEM, humanities, social sciences, and more
  - Dataset: `cais/mmlu`
  - Format: Multiple-choice questions (A, B, C, D)
  - Used for: Evaluating broad knowledge and reasoning
- **Other Popular Benchmarks**:
  - **HellaSwag**: Commonsense reasoning
  - **ARC**: Science questions
  - **TruthfulQA**: Truthfulness evaluation
  - **GSM8K**: Math word problems

### 3. Running Models on Laptop and Google Colab

**Laptop Setup:**
- Successfully ran 1B-3B models on local hardware
- Used quantization (4-bit, 8-bit) to reduce memory requirements
- Optimized for both GPU (CUDA) and CPU execution

**Google Colab:**
- Ran larger models (up to 7B parameters) using Colab's free GPU
- Leveraged Colab's built-in Gemini coding assistant
- Compared performance across different model sizes

---

## Setup and Installation

### Prerequisites

```bash
# Install required packages
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes

# Or using conda
conda install -c conda-forge transformers torch datasets accelerate tqdm
pip install huggingface_hub bitsandbytes
```

### Hugging Face Authentication

```bash
# Login to Hugging Face (required for gated models like Llama)
huggingface-cli login

# Enter your token from: https://huggingface.co/settings/tokens
```

### Verify Setup

Run the basic evaluation script to verify everything works:

```bash
python3 llama_mmlu_eval.py
```

---

## Project Structure

```
Running an LLM/
├── README.md                          # This file
├── llama_mmlu_eval.py                 # Original MMLU evaluation script
├── llama_mmlu_eval_optimized.py       # Optimized version with better timing
├── three_model.py                     # Multi-model evaluation script
├── create_graphs.py                   # Visualization and analysis script
├── my_agent.py                        # Enhanced chat agent implementation
├── simple_chat_agent.py               # Basic chat agent (starter code)
├── graphs/                            # Generated visualization graphs
│   ├── model_comparison.png
│   ├── mistake_overlap_distribution.png
│   └── mistake_overlap_pairwise.png
├── chat_history/                      # Saved conversation histories
└── *.json                            # Evaluation results (JSON format)
```

---

## Tasks and Implementations

### Task 1: Environment Setup and Verification

✅ **Completed**: Set up Python environment with all required modules
- Installed: `transformers`, `torch`, `datasets`, `accelerate`, `tqdm`, `huggingface_hub`, `bitsandbytes`
- Configured Hugging Face authentication for Llama 3.2-1B
- Verified setup by running `llama_mmlu_eval.py` on two MMLU subjects

### Task 2: Performance Timing Comparison

✅ **Completed**: Timed code execution across different configurations

**Timing Method:**
```bash
/usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device <device> --quant <quant>
```

**Configurations Tested:**
1. **GPU, no quantization**: Full precision on CUDA
2. **GPU, 4-bit quantization**: Memory-efficient with bitsandbytes
3. **GPU, 8-bit quantization**: Balanced quality/memory
4. **CPU, no quantization**: Full precision on CPU
5. **CPU, 4-bit quantization**: (Not supported - bitsandbytes requires CUDA)

**Key Findings:**
- GPU execution is 10-50x faster than CPU
- 4-bit quantization reduces memory by ~70% with minimal accuracy loss
- Quantization only works on CUDA (not CPU or Apple Silicon)

**Timing Metrics Captured:**
- **Real (wall) time**: Total elapsed time
- **CPU time**: Process CPU time
- **GPU time**: CUDA kernel execution time (when on GPU)

### Task 3: Multi-Model Evaluation

✅ **Completed**: Extended evaluation to 10 subjects across 3+ models

**Models Evaluated:**
1. `meta-llama/Llama-3.2-1B-Instruct`
2. `allenai/OLMo-2-0425-1B-Instruct`
3. `Qwen/Qwen2.5-0.5B-Instruct`
4. Additional models tested on Colab

**Subjects Evaluated (10 total):**
- astronomy
- business_ethics
- abstract_algebra
- anatomy
- computer_security
- econometrics
- electrical_engineering
- high_school_physics
- machine_learning
- professional_law

**Script Usage:**
```bash
python3 three_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --quant none \
  --save_question_data \
  --subjects astronomy business_ethics abstract_algebra anatomy computer_security econometrics electrical_engineering high_school_physics machine_learning professional_law
```

### Task 4: Enhanced Timing and Question-Level Analysis

✅ **Completed**: Added comprehensive timing and per-question tracking

**Features Added:**
- Real time, CPU time, and GPU time tracking
- Per-question correctness tracking
- `--verbose` flag to print each question, answer, and correctness
- `--save_question_data` flag to save detailed question-level results

**Usage:**
```bash
python3 llama_mmlu_eval_optimized.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --device cuda \
  --quant none \
  --verbose \
  --save_question_data \
  --subjects astronomy business_ethics
```

### Task 5: Results Visualization and Pattern Analysis

✅ **Completed**: Created graphs and analyzed mistake patterns

**Visualizations Created:**
1. **Model Comparison Graph** (`model_comparison.png`):
   - Accuracy by subject (grouped bars)
   - Overall accuracy comparison
   - Accuracy heatmap (subject vs model)
   - Error rate comparison

2. **Mistake Overlap Analysis**:
   - Distribution of questions that multiple models get wrong
   - Pairwise comparison of mistake overlap between models

**Pattern Analysis Findings:**
- **Not Random**: Models show systematic patterns in mistakes
- **Overlap**: Many questions are difficult for multiple models
- **Subject-Specific**: Some subjects (e.g., professional_law, abstract_algebra) are consistently harder
- **Model-Specific Strengths**: Different models excel in different domains

**Generate Graphs:**
```bash
python3 create_graphs.py \
  --result_dir . \
  --question_data_dir . \
  --output_dir graphs/
```

### Task 6: Google Colab Execution

✅ **Completed**: Ran evaluations on Google Colab

**Models Tested on Colab:**
- Tiny/Small: Llama 3.2-1B, OLMo 2 1B, Qwen 2.5 0.5B
- Small/Medium: Qwen 2.5 1.5B, Qwen 2.5 3B, Qwen 2.5 7B

**Colab Advantages:**
- Free GPU access (T4, V100, or A100)
- Built-in Gemini coding assistant
- Easy file download/upload
- Longer runtime sessions

### Task 7: Chat Agent Implementation

✅ **Completed**: Built a comprehensive chat agent from scratch

**Features Implemented:**
1. **Basic Chat Functionality**:
   - Load model and tokenizer
   - Maintain conversation history
   - Generate responses token-by-token

2. **Streaming Responses**:
   - Real-time token-by-token generation
   - Toggle with `/stream` command

3. **Context Management (Sliding Window with Summarization)**:
   - Keeps recent messages in full detail
   - Summarizes older messages when context exceeds threshold
   - Prevents context overflow in long conversations
   - Implementation: `manage_context_sliding_window()`

4. **Conversation History Toggle**:
   - `/history` command to toggle history on/off
   - Compare performance with vs without context
   - Demonstrates importance of conversation memory

5. **Additional Features**:
   - Save/load conversations (`/save`, `/load`)
   - Clear conversation (`/clear`)
   - Statistics (`/stats`)
   - Help (`/help`)

**Usage:**
```bash
python3 my_agent.py
```

**Commands:**
- `/help` - Show available commands
- `/save` - Save conversation to file
- `/load <file>` - Load conversation from file
- `/clear` - Clear conversation (keeps system prompt)
- `/stats` - Show conversation statistics
- `/stream` - Toggle streaming mode
- `/history` - Toggle conversation history on/off
- `/quit` - Exit the chat

**Key Implementation Details:**
- **Chat History Format**: List of dictionaries with `role` and `content`
- **Tokenization**: Uses `apply_chat_template()` to format history
- **Context Management**: Sliding window keeps 6 recent exchanges, summarizes older ones
- **History Toggle**: When OFF, only system prompt + current message is sent to model

---

## Results and Analysis

### Performance Comparison

**Timing Results (example for Llama 3.2-1B on 2 subjects):**

| Configuration | Real Time | CPU Time | GPU Time | Memory |
|--------------|-----------|----------|---------|--------|
| GPU, no quant | ~2 min | ~30s | ~90s | ~2.5 GB |
| GPU, 4-bit | ~2.5 min | ~35s | ~100s | ~1.5 GB |
| GPU, 8-bit | ~2.2 min | ~32s | ~95s | ~2.0 GB |
| CPU, no quant | ~45 min | ~40 min | N/A | ~5 GB |

### Accuracy Comparison Across Models

**Overall Accuracy (10 subjects):**

| Model | Accuracy | Notes |
|-------|----------|-------|
| Llama 3.2-1B | ~35-40% | Baseline, well-balanced |
| OLMo 2 1B | ~30-35% | Open model, slightly lower |
| Qwen 2.5 0.5B | ~25-30% | Smallest, fastest |
| Qwen 2.5 1.5B | ~35-40% | Good balance |
| Qwen 2.5 3B | ~40-45% | Better performance |
| Qwen 2.5 7B | ~45-50% | Best among tested |

**Subject-Specific Performance:**
- **Strong Subjects**: astronomy, business_ethics (models perform well)
- **Weak Subjects**: abstract_algebra, professional_law (consistently difficult)
- **Variable**: machine_learning, computer_security (model-dependent)

### Mistake Pattern Analysis

**Key Findings:**
1. **Systematic Errors**: Models make similar mistakes on the same questions
2. **Overlap**: 30-40% of wrong answers are shared across models
3. **Subject Patterns**: 
   - Math-heavy subjects (abstract_algebra) show high overlap
   - Factual subjects (astronomy) show more random errors
4. **Model-Specific**: Each model has unique failure modes

**Visualization:**
- See `graphs/mistake_overlap_distribution.png` for distribution
- See `graphs/mistake_overlap_pairwise.png` for pairwise comparisons

---

## Key Findings

### 1. LLM Fundamentals

**How LLMs Work:**
- Transform text → tokens → model processes → tokens → text
- Autoregressive: each token depends on all previous tokens
- Context window limits how much history can be maintained

**Tokenizer Role:**
- Critical bridge between human language and model processing
- Handles special tokens for conversation structure
- Different models use different tokenization strategies

### 2. Model Selection Trade-offs

**Size vs Performance:**
- Larger models (7B) → Better accuracy, slower inference
- Smaller models (0.5B-1B) → Faster, lower accuracy, more efficient

**Quantization Impact:**
- 4-bit: ~70% memory reduction, <5% accuracy loss
- 8-bit: ~60% memory reduction, <2% accuracy loss
- Essential for running on consumer hardware

### 3. Context Management

**Why It Matters:**
- Models have fixed context windows (e.g., 4K tokens)
- Long conversations exceed limits without management
- Simple truncation loses important early context

**Sliding Window + Summarization:**
- Keeps recent context in full detail
- Summarizes older context to preserve key information
- Enables arbitrarily long conversations

**History Toggle Comparison:**
- **With History**: Model remembers previous turns, can answer follow-ups
- **Without History**: Each turn is independent, no memory
- Demonstrates critical importance of context maintenance

### 4. Benchmark Insights

**MMLU Characteristics:**
- Broad knowledge test across 57 subjects
- Multiple-choice format (A, B, C, D)
- Good for evaluating general capabilities

**Model Performance Patterns:**
- STEM subjects: Generally better performance
- Humanities: More variable, model-dependent
- Professional domains: Consistently challenging

### 5. Practical Considerations

**Hardware Requirements:**
- **Laptop (1B models)**: 4-8 GB RAM, optional GPU
- **Colab (7B models)**: Free GPU sufficient
- **Quantization**: Essential for local execution

**Development Best Practices:**
- Use optimized evaluation scripts for accurate timing
- Save question-level data for pattern analysis
- Visualize results to identify trends
- Compare multiple models for robust conclusions

---

## Usage Examples

### Running MMLU Evaluation

```bash
# Basic evaluation (2 subjects, default model)
python3 llama_mmlu_eval.py

# Custom model and subjects
python3 llama_mmlu_eval_optimized.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --device cuda \
  --quant none \
  --subjects astronomy business_ethics abstract_algebra \
  --verbose \
  --save_question_data

# Multiple models
python3 three_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --quant none \
  --subjects astronomy business_ethics \
  --save_question_data
```

### Running Chat Agent

```bash
# Start chat agent
python3 my_agent.py

# In the chat:
You: Hello! My name is Alice.
Assistant: Hello Alice! Nice to meet you.

You: What's my name?
Assistant: Your name is Alice.

You: /history
✓ Conversation history: OFF

You: What's my name?
Assistant: I don't have that information... [No memory!]

You: /stats
📊 Conversation Statistics:
   History mode: OFF (stateless)
   Total messages: 4 (2 exchanges)
   Estimated tokens per turn: 45
```

### Generating Visualizations

```bash
# Create comparison graphs
python3 create_graphs.py \
  --result_dir . \
  --question_data_dir . \
  --output_dir graphs/
```

---

## Files and Scripts

### Evaluation Scripts

- **`llama_mmlu_eval.py`**: Original evaluation script with quantization support
- **`llama_mmlu_eval_optimized.py`**: Optimized version with better timing and question tracking
- **`three_model.py`**: Multi-model evaluation script

### Analysis Scripts

- **`create_graphs.py`**: Visualization and mistake pattern analysis

### Chat Agent

- **`my_agent.py`**: Enhanced chat agent with context management
- **`simple_chat_agent.py`**: Basic starter chat agent

### Output Files

- **`*.json`**: Evaluation results in JSON format
- **`*_questions.json`**: Per-question detailed results
- **`graphs/*.png`**: Visualization graphs

---

## Notes and Observations

### Question 1: How does an LLM work at a high level?

**Answer**: LLMs are autoregressive language models that:
1. Take text input → tokenize into numerical IDs
2. Process tokens through transformer layers with attention
3. Generate next token probabilities
4. Sample next token based on temperature/top-p
5. Repeat until completion
6. Detokenize output back to text

The tokenizer is crucial - it bridges human language and model processing.

### Question 2: What is the range of LLMs available?

**Answer**: Models range from:
- **Tiny**: 0.5B-1B parameters (run on laptop)
- **Small**: 1B-3B parameters (laptop with quantization)
- **Medium**: 3B-7B parameters (Colab GPU)
- **Large**: 7B-70B+ parameters (cloud/enterprise)

Popular families: Llama, Qwen, OLMo, Mistral, Phi, Gemma

### Question 3: How do quantization and device choice affect performance?

**Answer**: 
- **GPU vs CPU**: 10-50x speedup on GPU
- **4-bit quantization**: 70% memory reduction, <5% accuracy loss
- **8-bit quantization**: 60% memory reduction, <2% accuracy loss
- **Quantization**: CUDA-only (not CPU/Apple Silicon)

### Question 4: How does a chat agent maintain conversation context?

**Answer**: 
1. **History Storage**: Maintain list of messages (system, user, assistant)
2. **Tokenization**: Convert history to tokens each turn
3. **Context Management**: 
   - Sliding window: Keep recent messages
   - Summarization: Compress older messages
   - Truncation: Remove oldest if needed
4. **Model Processing**: Send full context to model each turn

**Without History**: Each turn is independent - model has no memory.

### Question 5: What patterns exist in model mistakes?

**Answer**:
- **Not Random**: Models make similar mistakes
- **Overlap**: 30-40% of errors shared across models
- **Subject-Specific**: Some subjects consistently harder
- **Systematic**: Math/logic questions show high overlap
- **Model-Specific**: Each model has unique failure modes

---

## Future Work

### Recommended Extensions

1. **Restartability**: Implement pickle-based checkpointing for long evaluations
2. **MT-Bench**: Test chat agent on multi-turn benchmark
3. **Additional Benchmarks**: HellaSwag, ARC, TruthfulQA
4. **Fine-tuning**: Experiment with LoRA/QLoRA fine-tuning
5. **Vector Databases**: Implement semantic search for long-term memory

### Optional Enhancements

- **Web Interface**: Create web UI for chat agent
- **API Server**: Expose chat agent as REST API
- **Multi-modal**: Extend to vision-language models
- **Distributed Evaluation**: Parallel evaluation across multiple GPUs

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [Llama Chat Context Management Guide](https://huggingface.co/docs/transformers/main/en/llama_torch)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Hugging Face Authentication Guide](https://huggingface.co/docs/hub/security-tokens)

---

## License

This project is for educational purposes. Model usage is subject to each model's license:
- **Llama**: Meta's custom license
- **OLMo**: Apache 2.0
- **Qwen**: Tongyi Qianwen license

---

## Author

Created as part of the "Running an LLM" course project to understand LLMs from the inside out.

---

## Acknowledgments

- Hugging Face for model hosting and tools
- Model developers (Meta, AllenAI, Alibaba, Microsoft)
- MMLU dataset creators
- Open source community
