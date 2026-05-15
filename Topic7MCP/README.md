# Topic7MCP

This repo has two small demo projects:

- MCP examples that talk to the Asta / Semantic Scholar MCP server
- A2A examples that let agents register, find each other, and exchange tasks

The code is meant for learning and classroom demos. Some files are complete apps, and some are simple starter templates.

## Setup

Install the Python packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file for the scripts that use OpenAI or Asta:

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

## Main Files

### MCP scripts

- `mcp_list_tools.py`
  Lists the tools exposed by the Asta MCP server and prints the required and optional fields for each one.

- `mcp_search_recent_agent_papers.py`
  Searches for papers about large language model agents, then fetches full details for each result.

- `mcp_recent_citations.py`
  Gets recent citation data for one paper ID and prints a short list of citing papers.

- `mcp_reference_timeline.py`
  Pulls a paper's references and sorts them by year so you can see the older foundation first.

- `mcp_research_chatbot.py`
  A Gradio chatbot that uses LangGraph plus MCP tools for paper search and paper lookup.

- `mcp_citation_network_report.py`
  Builds a small research pipeline around one paper. It gathers the seed paper, top references, recent citations, and author profiles, then asks OpenAI to turn that data into a markdown report.

### A2A scripts

- `a2a_agent_server_template.py`
  A full FastAPI template for an A2A-style agent. It exposes an agent card, accepts tasks, finds its ngrok URL, and tries to register with the registry on startup.

- `a2a_agent_registry.py`
  A simple registry server for agents. It supports registration, listing, filtering by skill, direct send, broadcast, health checks, reset, and a small HTML dashboard.

- `a2a_local_system_test.py`
  Starts a local registry plus fake agents and runs an end-to-end test of the A2A flow.

- `a2a_trivia_tournament.py`
  Runs a trivia game across registered agents. It can score answers with OpenAI and can route questions to the best-matching agents instead of broadcasting to everyone.

- `a2a_minimal_task_app.py`
  A very small FastAPI sketch for a `/task` endpoint. It is not a full app yet because `call_my_llm(question)` is still a placeholder.

### Helper script

- `check_ngrok_tunnel.sh`
  Checks whether ngrok can open a tunnel from the current network. Useful before class or before testing public A2A agents.

## Quick Start

### MCP examples

Run any of these after setting `ASTA_API_KEY`:

```bash
python mcp_list_tools.py
python mcp_search_recent_agent_papers.py
python mcp_recent_citations.py
python mcp_reference_timeline.py
```

To launch the chatbot:

```bash
python mcp_research_chatbot.py
```

To generate a paper report:

```bash
python mcp_citation_network_report.py ARXIV:2210.03629
python mcp_citation_network_report.py "retrieval augmented generation"
```

### A2A local test

If you want to test the A2A flow without ngrok or real student agents:

```bash
python a2a_local_system_test.py
```

### A2A registry and agent

Start the registry:

```bash
python a2a_agent_registry.py
```

Start ngrok in another terminal:

```bash
ngrok http 8000
```

Then start the agent template:

```bash
python a2a_agent_server_template.py
```

For local prompt testing without ngrok or the registry:

```bash
python a2a_agent_server_template.py --dryrun
```

### Trivia tournament

Once agents are registered, run:

```bash
python a2a_trivia_tournament.py
```

Useful options:

- `--no-score` skips OpenAI judging
- `--funny` gives a bonus point for the funniest wrong answer
- `--smart-route` sends each question only to the best-matching agents
- `--top N` sets how many agents are used in smart-route mode
- `--pause` waits after each question

Example:

```bash
python a2a_trivia_tournament.py --smart-route --top 2 --funny
```

## Notes

- The default agent template in this repo is a science-themed persona. It answers science questions directly and gives intentionally silly answers on non-science topics.
- The chatbot export file is saved as `mcp_research_chat_export.txt`.
- There are also sample output files in the repo from earlier runs of the MCP scripts.
