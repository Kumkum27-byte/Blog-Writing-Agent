# Blog Writing Agent 🤖✍️

A production-grade **multi-agent content pipeline** built with LangGraph that autonomously researches, plans, writes, and illustrates technical blog posts — end to end.

## What it does

You give it a topic. It figures out whether it needs to search the web, plans a structured outline, writes each section in parallel, decides where images go, and generates them — all without you touching a thing.

**Pipeline flow:**

```
Topic Input
    ↓
Router Node — closed_book / hybrid / open_book?
    ↓ (if research needed)
Research Node — Tavily web search + evidence synthesis
    ↓
Orchestrator Node — structured blog plan (5–7 sections)
    ↓
Worker Nodes — parallel section writing (fan-out via Send)
    ↓
Reducer Subgraph
    ├── merge_content — ordered assembly
    ├── decide_images — placement + prompt generation
    └── generate_and_place_images — Gemini image gen + MD injection
    ↓
Final Markdown Blog saved to output_blogs/
```

## Key Features

- **Intelligent routing** — classifies topics into `closed_book`, `hybrid`, or `open_book` mode; triggers live web research only when needed, reducing unnecessary API calls
- **Parallel section writing** — uses LangGraph's `Send`-based fan-out so all sections are written concurrently, not sequentially
- **Strict Pydantic schemas** — every pipeline stage (Plan, Task, EvidencePack, ImageSpec) enforces structured outputs, preventing silent failures
- **AI image generation** — Gemini generates contextual diagrams and inserts them at the right sections automatically
- **Streamlit UI** — tabbed interface showing Plan, Evidence, Markdown Preview, Images, and Logs; includes past blog loader and download options (MD + image bundle ZIP)

## Tech Stack

| Layer | Tools |
|---|---|
| Agent framework | LangGraph (StateGraph, Send, subgraph) |
| LLM | Google Gemini 2.5 Flash |
| Web research | Tavily Search API |
| Structured outputs | Pydantic v2 |
| Frontend | Streamlit |
| Image generation | Gemini image model |

## Project Structure

```
Blog-Writing-Agent/
├── bwa_backend.py      # LangGraph pipeline — all nodes and graph compilation
├── bwa_frontend.py     # Streamlit UI
├── output_blogs/       # Generated markdown files saved here
├── images/             # AI-generated images saved here
├── requirements.txt
├── .env                # API keys (not committed)
└── .gitignore
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Kumkum27-byte/Blog-Writing-Agent
cd Blog-Writing-Agent
```

**2. Create virtual environment and install dependencies**
```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Get your keys:
- Gemini API key → [Google AI Studio](https://aistudio.google.com/)
- Tavily API key → [Tavily](https://tavily.com/) (free tier available)

**4. Run the app**
```bash
streamlit run bwa_frontend.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. Enter a topic in the sidebar (e.g. *"How transformers work"* or *"Latest trends in RAG 2025"*)
2. Click **Generate Blog**
3. Watch the pipeline run — nodes stream live in the status panel
4. View the result across tabs: Plan → Evidence → Preview → Images
5. Download the markdown or the full bundle (MD + images ZIP)

Previously generated blogs appear in the sidebar under **Past blogs** — click any to reload it.

## Technical Decisions Worth Noting

**Why LangGraph over a simple chain?**
Parallel section writing via `Send`-based fan-out means a 6-section blog writes all sections concurrently instead of sequentially — significant latency reduction.

**Why structured Pydantic outputs everywhere?**
Early versions used raw LLM text; silent truncation bugs caused entire sections to disappear. Strict schemas catch failures at parse time rather than silently corrupting output.

**Biggest bug fixed:**
`decide_images` originally asked the LLM to regenerate the full blog markdown inside a structured output field — causing silent truncation on long blogs. Fixed by separating image decision (placeholders only) from image generation (deterministic code-side insertion).

## Author

**Kumkum** — [GitHub](https://github.com/Kumkum27-byte) · [LinkedIn](https://linkedin.com/in/kumkummorewal-b9291a278)
