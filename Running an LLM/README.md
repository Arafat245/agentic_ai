# Running an LLM

**Topic 1** — Agentic AI Spring 2026 | [Course Page](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/Running_an_LLM.html)

This repository contains implementations and experiments for understanding Large Language Models (LLMs) from the ground up, including model evaluation, chat agent development, and performance analysis.

## Portfolio Contents

Per course requirements, this subdirectory includes:
- **Code**: All evaluation scripts, chat agent, and graph generation tools
- **Graphs (PDF)**: Accuracy and performance comparison across multiple models and benchmark datasets (MMLU, ARC)
- **Notes**: This markdown file discussing the questions from the tasks

## Table of Contents

- [Learning Goals](#learning-goals)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Tasks and Implementations](#tasks-and-implementations)
- [Results and Analysis](#results-and-analysis)
- [Key Findings](#key-findings)
- [Optional Tasks](#optional-tasks)
- [References](#references)

---

## Learning Goals

*(From [course instructions](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/Running_an_LLM.html))*

- Understand how an LLM works at a high level and the role of a tokenizer
- Have knowledge of the range of LLMs available and the major LLM benchmark datasets on Hugging Face
- Be able to run tiny/small LLMs on your laptop and small/medium ones on Google Colab on a variety of benchmarks
- Understand how a chat agent maintains the conversation context, and be able to build a simple chat agent

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

### Task 1: Prerequisites

Per [course instructions](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/Running_an_LLM.html):

```bash
# Required packages (conda or pip)
pip install -r requirements.txt

# Or install manually:
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes
pip install matplotlib seaborn numpy   # for graph scripts
```

### Task 2: Hugging Face Authentication

Set up [Hugging Face authorization](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/COMPLETE_HF_AUTH_GUIDE.html) for Llama 3.2-1B:

```bash
huggingface-cli login
# Enter token from: https://huggingface.co/settings/tokens
```

### Task 3: Verify Setup

Run [llama_mmlu_eval.py](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/llama_mmlu_eval.py) on two [MMLU topics](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/datasets2025.html):

```bash
conda activate agentic_ai   # or your env with dependencies
python3 llama_mmlu_eval.py
```

---

## Project Structure

```
Running an LLM/
├── README.md                          # This file (notes discussing task questions)
├── requirements.txt                   # Python dependencies
├── llama_mmlu_eval.py                 # Original MMLU evaluation script
├── llama_mmlu_eval_optimized.py       # Optimized version with better timing
├── three_model.py                     # Multi-model evaluation script
├── arc_eval.py                        # ARC benchmark evaluation script
├── create_graphs.py                   # MMLU visualization (PDF/PNG)
├── create_cross_dataset_graphs.py     # Cross-dataset comparison (MMLU + ARC)
├── my_agent.py                        # Enhanced chat agent implementation
├── simple_chat_agent.py               # Basic chat agent (starter code)
├── graphs/                            # Generated graphs (PDF for portfolio)
│   ├── model_comparison.pdf
│   ├── mistake_overlap_distribution.pdf
│   ├── mistake_overlap_pairwise.pdf
│   └── cross_dataset_comparison.pdf   # MMLU vs ARC comparison
├── chat_history/                      # Saved conversation histories
└── *.json                            # Evaluation results (JSON format)
```

---

## Tasks and Implementations

### Task 1–3: Environment Setup and Verification

✅ **Completed**: Set up Python environment with all required modules
- Installed: `transformers`, `torch`, `datasets`, `accelerate`, `tqdm`, `huggingface_hub`, `bitsandbytes`, `matplotlib`, `seaborn`, `numpy`
- Configured [Hugging Face authentication](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/COMPLETE_HF_AUTH_GUIDE.html) for Llama 3.2-1B
- Verified setup by running `llama_mmlu_eval.py` on two MMLU subjects

### Task 4: Performance Timing Comparison

✅ **Completed**: Timed code execution across different configurations

**Timing Method** (using `time` shell command):
```bash
/usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant none
/usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant 4bit
/usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant 8bit
/usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cpu --quant none
# CPU + 4-bit: Not possible (bitsandbytes requires CUDA; skip on MacBook)
```

**Configurations Tested:**
1. **GPU, no quantization**: Full precision on CUDA
2. **GPU, 4-bit quantization**: Memory-efficient with bitsandbytes *(skip on MacBook)*
3. **GPU, 8-bit quantization**: Balanced quality/memory *(skip on MacBook)*
4. **CPU, no quantization**: Full precision on CPU
5. **CPU, 4-bit quantization**: Not supported (bitsandbytes requires CUDA)

**Key Findings:**
- GPU execution is 10-50x faster than CPU
- 4-bit quantization reduces memory by ~70% with minimal accuracy loss
- Quantization only works on CUDA (not CPU or Apple Silicon)

**Timing Metrics Captured:**
- **Real (wall) time**: Total elapsed time
- **CPU time**: Process CPU time
- **GPU time**: CUDA kernel execution time (when on GPU)

### Task 5: Multi-Model Evaluation (Code Modifications)

✅ **Completed**: Modified code per course requirements

**5a. Run on 10 subjects using 2 other tiny/small models** (in addition to Llama 3.2-1B):
- `meta-llama/Llama-3.2-1B-Instruct`
- `allenai/OLMo-2-0425-1B-Instruct` — [LLM reference](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/llm_vlm_models_reference.html)
- `Qwen/Qwen2.5-0.5B-Instruct`
- Additional models tested on Colab

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

**5b. Add timing information** to evaluation summary: real time, CPU time, GPU time

**5c. Add option to print** each question, model answer, and correctness:
- `--verbose` flag
- `--save_question_data` for detailed question-level results

**Example:**
```bash
python3 llama_mmlu_eval_optimized.py --model allenai/OLMo-2-0425-1B-Instruct --device cuda --quant none --verbose --save_question_data --subjects astronomy business_ethics
```

### Task 6: Results Visualization and Pattern Analysis

✅ **Completed**: Ran code and created graphs; analyzed mistake patterns

**Course questions:**
- *Can you see any patterns to the mistakes each model makes or do they appear random?* → See [Mistake Pattern Analysis](#mistake-pattern-analysis)
- *Do all the models make mistakes on the same questions?* → Yes, significant overlap; see pairwise graphs

**Visualizations Created:**
1. **Model Comparison Graph** (`graphs/model_comparison.pdf`):
   - Accuracy by subject (grouped bars)
   - Overall accuracy comparison
   - Accuracy heatmap (subject vs model)
   - Error rate comparison
   - *See full graph in [Results and Analysis](#results-and-analysis) section*

2. **Mistake Overlap Analysis**:
   - **Distribution Graph** (`graphs/mistake_overlap_distribution.pdf`): Shows how many questions are answered incorrectly by multiple models
   - **Pairwise Comparison** (`graphs/mistake_overlap_pairwise.pdf`): Detailed comparison of mistake overlap between model pairs
   - *See full graphs in [Mistake Pattern Analysis](#mistake-pattern-analysis) section*

**Pattern Analysis Findings:**
- **Not Random**: Models show systematic patterns in mistakes
- **Overlap**: Many questions are difficult for multiple models
- **Subject-Specific**: Some subjects (e.g., professional_law, abstract_algebra) are consistently harder
- **Model-Specific Strengths**: Different models excel in different domains

**Generate Graphs** (PDF format for portfolio):
```bash
python3 create_graphs.py --result_dir . --question_data_dir . --output_dir graphs/ --format pdf
```
*Requires: `pip install matplotlib seaborn numpy`*

**Graph Output:**
All graphs are saved in PDF format (portfolio requirement) in the `graphs/` directory:
- `model_comparison.pdf` - Comprehensive model performance comparison
- `mistake_overlap_distribution.pdf` - Distribution of mistake overlap
- `mistake_overlap_pairwise.pdf` - Pairwise model mistake comparison
- `cross_dataset_comparison.pdf` - Model performance across MMLU and ARC benchmarks

### Task 7: Google Colab Execution

✅ **Completed**: Repeated steps on [Google Colab](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/COLAB_MMLU_GUIDE.html)

**Models Tested on Colab:**
- **3 tiny/small**: Llama 3.2-1B, OLMo 2 1B, Qwen 2.5 0.5B
- **3 small/medium**: Qwen 2.5 1.5B, Qwen 2.5 3B, Qwen 2.5 7B — [reference](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/llm_vlm_models_reference.html)

**Colab advantages:** Free GPU, built-in Gemini coding assistant, file download/upload

### Multiple Benchmark Datasets (Portfolio Requirement)

✅ **Completed**: Added ARC benchmark for cross-dataset comparison

**ARC (AI2 Reasoning Challenge):**
- Science reasoning multiple-choice dataset
- Configs: ARC-Easy, ARC-Challenge
- Dataset: `allenai/ai2_arc` on Hugging Face

**Usage:**
```bash
# Evaluate on ARC-Easy (quick test with 100 examples)
python3 arc_eval.py --model meta-llama/Llama-3.2-1B-Instruct --config ARC-Easy --max_examples 100

# Full ARC-Challenge evaluation
python3 arc_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --config ARC-Challenge --device cuda
```

**Cross-Dataset Graphs:**
```bash
# Generate PDF comparing models across MMLU and ARC
python3 create_graphs.py --result_dir . --question_data_dir . --output_dir graphs/ --format pdf
python3 create_cross_dataset_graphs.py --result_dir . --output_dir graphs/ --format pdf
```

### Task 8: Chat Agent Implementation

✅ **Completed**: Built chat agent from scratch (no pre-defined chat library)

Based on [simple chat agent](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/simple_chat_agent.py); implements [context management](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/CONTEXT_MANAGEMENT.html)

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

4. **Conversation History Toggle** (course requirement):
   - `/history` command to turn off conversation history
   - Compare chat agent on multi-turn conversation: with history maintained vs. when it is not
   - With history OFF: each turn sees only system prompt + current message (no memory)

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

**Model Comparison Visualization:**

![Model Comparison](graphs/model_comparison.pdf)

*This comprehensive comparison shows:*
- *Top-left: Accuracy by subject (grouped bars for each model)*
- *Top-right: Overall accuracy comparison across all models*
- *Bottom-left: Accuracy heatmap showing subject vs model performance*
- *Bottom-right: Error rate comparison*

### Mistake Pattern Analysis

**Key Findings:**
1. **Systematic Errors**: Models make similar mistakes on the same questions
2. **Overlap**: 30-40% of wrong answers are shared across models
3. **Subject Patterns**: 
   - Math-heavy subjects (abstract_algebra) show high overlap
   - Factual subjects (astronomy) show more random errors
4. **Model-Specific**: Each model has unique failure modes

**Mistake Overlap Distribution:**

This graph shows how many questions are answered incorrectly by multiple models. The distribution reveals whether mistakes are random or systematic.

![Mistake Overlap Distribution](graphs/mistake_overlap_distribution.pdf)

*Key insights from the distribution:*
- *Questions that all models get wrong indicate genuinely difficult questions*
- *Questions that only one model gets wrong suggest model-specific weaknesses*
- *High overlap (multiple models wrong) indicates systematic challenges*

**Pairwise Mistake Overlap:**

This detailed comparison shows mistake overlap between pairs of models, helping identify which models make similar errors and which have unique failure patterns.

![Pairwise Mistake Overlap](graphs/mistake_overlap_pairwise.pdf)

*Analysis of pairwise comparisons:*
- *"Both wrong" bars show questions where both models failed*
- *"Only first wrong" / "Only second wrong" show model-specific errors*
- *High "both wrong" indicates shared difficulty areas*
- *High "only X wrong" indicates unique model weaknesses*

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
# Create comparison graphs (PDF for portfolio)
python3 create_graphs.py \
  --result_dir . \
  --question_data_dir . \
  --output_dir graphs/ \
  --format pdf

# Create cross-dataset comparison (MMLU + ARC)
python3 create_cross_dataset_graphs.py --result_dir . --output_dir graphs/ --format pdf
```

---

## Files and Scripts

### Evaluation Scripts

- **`llama_mmlu_eval.py`**: Original MMLU evaluation (2 subjects)
- **`llama_mmlu_eval_optimized.py`**: Optimized with timing, `--verbose`, `--save_question_data`
- **`three_model.py`**: Multi-model MMLU evaluation
- **`arc_eval.py`**: ARC benchmark evaluation (ARC-Easy, ARC-Challenge)

### Analysis Scripts

- **`create_graphs.py`**: MMLU visualization and mistake pattern analysis (PDF/PNG)
- **`create_cross_dataset_graphs.py`**: Cross-dataset comparison (MMLU + ARC)

### Chat Agent

- **`my_agent.py`**: Enhanced chat agent with context management
- **`simple_chat_agent.py`**: Basic starter chat agent

### Output Files

- **`*.json`**: Evaluation results in JSON format
- **`*_questions.json`**: Per-question detailed results
- **`graphs/*.pdf`**: Visualization graphs (PDF for portfolio submission)

---

## Notes and Observations

*Discussion of questions from the tasks (portfolio requirement)*

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

## Optional Tasks

### Task 9 (Recommended): Restartability

*Learn how to make the program restartable if killed using the [pickle library](https://docs.python.org/3/library/pickle.html).* — Not yet implemented; listed in Future Work.

### Task 10 (Ambitious): MT-Bench

*Get [MT-Bench](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/MULTITURN_BENCHMARKS_GUIDE.html) installed and test chat agent; compare with other students' agents.* — Not yet implemented.

---

## Future Work

- **Restartability**: Pickle-based checkpointing for long evaluations (Task 9)
- **MT-Bench**: Multi-turn benchmark with GPT-4 judge (Task 10)
- **Additional Benchmarks**: HellaSwag, TruthfulQA (ARC implemented)
- **Fine-tuning**: LoRA/QLoRA experiments
- **Vector Databases**: Semantic search for long-term memory

---

## References

### Course Resources

- [Topic 1: Running an LLM](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/Running_an_LLM.html)
- [LLM and VLM Reference Guide](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/llm_vlm_models_reference.html)
- [Datasets](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/datasets2025.html)
- [Hugging Face Authentication Guide](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/COMPLETE_HF_AUTH_GUIDE.html)
- [Guide to Running llama_mmlu_eval on Google Colab](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/COLAB_MMLU_GUIDE.html)
- [Llama Chat Context Management Guide](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/CONTEXT_MANAGEMENT.html)
- [Multi-Turn Benchmark Guide](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/MULTITURN_BENCHMARKS_GUIDE.html)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

### External

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [ARC Dataset](https://huggingface.co/datasets/allenai/ai2_arc)

---

## License

This project is for educational purposes. Model usage is subject to each model's license:
- **Llama**: Meta's custom license
- **OLMo**: Apache 2.0
- **Qwen**: Tongyi Qianwen license

---

## Author

Created as part of **Topic 1: Running an LLM** — Agentic AI Spring 2026. Subdirectory named "Running an LLM" per [portfolio requirements](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running/Running_an_LLM.html).

---

## Acknowledgments

- Hugging Face for model hosting and tools
- Model developers (Meta, AllenAI, Alibaba, Microsoft)
- MMLU dataset creators
- Open source community
