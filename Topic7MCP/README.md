# Topic7MCP

This repo has two demo parts:

- MCP exercises that use the Asta / Semantic Scholar MCP server
- A2A examples for agent registration, routing, and task passing

The MCP files now keep the original exercise labels in the filenames so it is easy to match the code with the assignment.

## Setup

Install packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file for the scripts that need keys:

```env
ASTA_API_KEY=your_asta_key
OPENAI_API_KEY=your_openai_key
```

Optional A2A settings:

```env
REGISTRY_URL=http://localhost:8001
LLM_MODEL=gpt-4o-mini
PORT=8000
```

## MCP Exercises

### Exercise A

File: `mcp_exA_list_tools.py`

Purpose: asks the MCP server for the tool list and prints a short summary for each tool.

Run:

```bash
python mcp_exA_list_tools.py
```

Example output:

```text
Tool: get_paper
  Description: Get details about a paper by its id.
  Required: paper_id (string)
  Optional: fields (string)

Tool: get_paper_batch
  Description: Get details about a list of papers by their ids.
  Required: ids (array)
  Optional: fields (string)
```

Saved sample output: `mcp_exA_list_tools_output.txt`

### Exercise B

#### Drill 1

File: `mcp_exB_drill1_search_recent_agent_papers.py`

Purpose: searches for recent papers on large language model agents and then fetches the full details for each match.

Run:

```bash
python mcp_exB_drill1_search_recent_agent_papers.py
```

Example output:

```text
Title:    InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents
Year:     2024
Authors:  Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang
Abstract: Recent work has embodied LLMs as agents...
```

Saved sample output: `mcp_exB_drill1_search_recent_agent_papers_output.txt`

#### Drill 2

File: `mcp_exB_drill2_recent_citations.py`

Purpose: looks up recent citations for one paper ID. The current file uses `ARXIV:1810.04805`.

Run:

```bash
python mcp_exB_drill2_recent_citations.py
```

Example output:

```text
Querying the last 10 citations for paper ID: ARXIV:1810.04805...

Total citations returned: 10

First 5 citing papers:
  - Chuanyang Gong et al. Enhancing large language models for knowledge graph question answering... (2026)
  - Jiapei Hu et al. Multi-view dynamic perception framework for Chinese harmful meme detection (2026)
```

Saved sample output: `mcp_exB_drill2_recent_citations_output.txt`

#### Drill 3

File: `mcp_exB_drill3_reference_timeline.py`

Purpose: gets the references for one paper, sorts them by year, and prints a simple timeline. The current file also uses `ARXIV:1810.04805`.

Run:

```bash
python mcp_exB_drill3_reference_timeline.py
```

Example output:

```text
Querying references for paper ID: ARXIV:1810.04805...

Total references: 39

References sorted by year (ascending):
  1950  Computing Machinery and Intelligence
  1998  BERT-related earlier work example...
```

Saved sample output: `mcp_exB_drill3_reference_timeline_output.txt`

### Exercise C

File: `mcp_exC_research_chatbot.py`

Purpose: runs a Gradio chatbot backed by LangGraph, OpenAI, and MCP tools.

Run:

```bash
python mcp_exC_research_chatbot.py
```

Example user inputs:

```text
Find recent papers about large language model agents
Who wrote Attention is All You Need and what else have they published?
What papers cite the original BERT paper?
Summarize the references used in the ReAct paper
```

Example chatbot output:

```text
The paper "Attention is All You Need" was written by:
- Ashish Vaswani
- Noam Shazeer
- Niki Parmar
- Jakob Uszkoreit
...
```

Saved chat export file: `mcp_exC_research_chat_export.txt`

### Exercise D

File: `mcp_exD_citation_network_report.py`

Purpose: builds a small research pipeline around one seed paper and then writes a markdown report.

Run by paper ID:

```bash
python mcp_exD_citation_network_report.py ARXIV:2210.03629
```

Run by keyword:

```bash
python mcp_exD_citation_network_report.py "retrieval augmented generation"
```

Example report sections:

```text
1. Summary of the seed paper
2. Foundational Works
3. Recent Developments
4. Author Profiles
5. Research Gaps
```

Saved sample reports:

- `mcp_exD_citation_network_report_output.md`
- `mcp_exD_citation_network_report_output_2.md`

## A2A Files

- `a2a_agent_server_template.py`
  Full FastAPI agent template with agent card, task endpoint, ngrok detection, and registry registration.

- `a2a_agent_registry.py`
  Registry server with register, list, filter, send, broadcast, health, reset, and dashboard endpoints.

- `a2a_local_system_test.py`
  Starts a local registry plus fake agents and runs an end-to-end A2A test.

- `a2a_trivia_tournament.py`
  Runs a trivia game across registered agents. It supports scoring and simple smart routing.

- `a2a_minimal_task_app.py`
  Very small starter app for a `/task` endpoint.

## A2A Quick Start

Local test:

```bash
python a2a_local_system_test.py
```

Start the registry:

```bash
python a2a_agent_registry.py
```

Start ngrok in another terminal:

```bash
ngrok http 8000
```

Start the agent template:

```bash
python a2a_agent_server_template.py
```

Dry run mode:

```bash
python a2a_agent_server_template.py --dryrun
```

Trivia tournament:

```bash
python a2a_trivia_tournament.py
python a2a_trivia_tournament.py --smart-route --top 2 --funny
```

## Helper Script

`check_ngrok_tunnel.sh`

Use this before class if you want to confirm ngrok works on the current network.

```bash
chmod +x check_ngrok_tunnel.sh
./check_ngrok_tunnel.sh
```

## Notes

- The default A2A agent in this repo is a science-themed persona.
- Some example inputs, paper IDs, and limits are hard-coded in the exercise scripts. You can edit them directly in the file if needed.
