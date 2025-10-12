# Multi-Agent Research System

A LangGraph + LLM prototype that demonstrates a multi-agent research workflow (Supervisor → Researcher → Analyst → Writer) with tool integrations and a simple Streamlit demo.

This repository contains a notebook exploration and a Streamlit app that programmatically constructs the same agentic StateGraph. The system shows how to combine agent states, tool nodes, human-in-the-loop interrupts, and a final writer that outputs a consolidated report.

Highlights
- Supervisor-led multi-agent workflow: supervisor decides which agent runs next (researcher, analyst, writer) and the system loops until the task is complete.
- Tools: web search (Tavily), arXiv search, summarizer/writer, and a human assistance interrupt that pauses the run and accepts a `Command(resume=...)` to continue.
- Streamlit demo (`app.py`) reproduces the notebook's agentic flow and logs which tools executed.
- Loop-safety: the researcher includes a `tool_round` counter and `MAX_TOOL_ROUNDS` guard to avoid infinite researcher↔tool loops.

Project layout
- `Langgraph/app.py` — Streamlit demo that constructs and compiles the LangGraph StateGraph and exposes a minimal UI to run queries.
- `4.langgraph/notebooks/MultiAgent.ipynb` — Notebook with the supervisor, researcher, analyst, writer agents, tool wrappers, and human-in-the-loop cells (interrupt/resume helpers).
- `.env` — local environment variables (do not commit). This repository currently contains sensitive keys in `.env` — see the Security section below.

Quickstart (local)

1. Create a virtual environment and install dependencies. If the project doesn't provide `requirements.txt`, install the packages used in the notebook and app (LangGraph, LangChain bindings, Groq client, Streamlit, etc.).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present; otherwise install packages manually
```

2. Provide credentials via environment variables (do NOT commit these files). Example variables used by this project:

- GROQ_API_KEY
- TAVILY_API_KEY
- GOOGLE_API_KEY (optional)
- LANGCHAIN_API_KEY (optional)
- LANGSMITH_ENDPOINT / LANGSMITH_PROJECT (optional)

You can export them in your shell for local testing (replace with your real, rotated keys):

```bash
export GROQ_API_KEY="your_groq_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
# or create a local .env file and load it in your shell (do NOT commit .env)
```

3. Run the Streamlit demo:

```bash
streamlit run app.py
```

4. Open `4.MultiAgent.ipynb` in Jupyter or VS Code to explore the multi-agent examples and the human-in-the-loop cells. The notebook contains helper functions `cancel_interrupted_run()` and `compile_shutdown_workflow()` which are useful when experimenting with interruptions.

Notes and troubleshooting

- "Tools should have a name!" or "Function must have a docstring" errors: The LangGraph/LLM tool conversion step may require each tool to expose metadata. Make sure functions you pass into `ToolNode` or `llm.bind_tools([...])` are decorated or wrapped with explicit tool metadata (for example `@tool(name="search_web", description="Search the web for relevant results")`) or created as structured tool objects.

- If `app.py` fails to import: run a quick syntax/type check and examine the stack trace. In many cases the error stems from missing imports, unavailable environment variables, or tools lacking metadata.

- Streaming/resume behavior: Interrupt/resume via `Command(resume=...)` requires a compiled workflow with a persistent checkpointer (for example a `MemorySaver` or a persistent storage backend) if you expect resume commands to come from a different process or after a long pause.

Security & secrets (important)

This repository currently contains a `.env` file with API keys. Secrets must never be committed. If you accidentally committed keys, follow these steps **immediately**:

1. Revoke or rotate the exposed keys via the provider dashboards (Groq, Tavily, Google Cloud, LangChain, etc.). Do this before attempting to clean the history.

2. Remove the `.env` from the working tree and add it to `.gitignore`:

```bash
# remove from index, keep local file
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Remove .env from index and ignore it"
```

3. If the secret is present in commit history (not just the working tree), you must purge it from history. Two common tools are `git-filter-repo` (recommended) and BFG Repo-Cleaner. Example (git-filter-repo):

```bash
# install git-filter-repo first (follow official instructions)
# make a backup clone before running destructive history rewrite
git clone --mirror git@github.com:youruser/yourrepo.git repo-mirror.git
cd repo-mirror.git
# remove the file from all commits
git filter-repo --invert-paths --paths .env
# push the cleaned history (force push)
git push --force
```

Warning: rewriting history requires force-pushing and will affect collaborators — coordinate before doing this.

4. Verify GitHub's secret scanning and security alerts: the repository may still block pushes until the secret is either rotated or removed from history. Use the GitHub Security -> Secret scanning UI to review blocked secrets.

Developer notes

- The Streamlit app constructs and compiles the graph at import time. For deployment, consider compiling once at startup and caching the compiled graph to avoid repeated cold-compilation.
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs. If you want, I can generate a minimal `requirements.txt` from the notebook/app imports.
- Tests: add a small smoke test that imports `app.py`, compiles the graph, and invokes it with a fake LLM or stubbed tools to validate the wiring without calling external APIs.

Contributing

If you'd like help hardening the app for deployment, cleaning secrets, or writing tests, say which you'd like me to do next and I will proceed.

License

Add your preferred open-source license here.

Contact

If you want me to also create a small `requirements.txt`, CI/test harness, or a secure `.env.example` (with environment variable names but no secrets), tell me which one to create next.
