from __future__ import annotations

import os
import operator
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash',temperature=0.7)


#SCHEMAS

class Task(BaseModel):
    id : int
    title : str
    goal : str = Field(..., description = 'One sentence describing what the readder should be able to do/understand after this section')
    bullets : list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description = '3-5 concrete subpoints to cover in this section'
    )
    target_words : int = Field(
        ...,
        description = 'target word count for this section(100-200)'
    )
    tags : list[str] = Field(default_factory=list)
    requires_research : bool = False
    requires_citations : bool = False
    requires_code : bool = False
    

class Plan(BaseModel):
    blog_title : str
    audience : str = Field(..., description = 'Who is this blog for.')
    tone : str = Field(..., description = 'Writing tone(eg, futuristic, practical, neutral)')
    tasks : list[Task]


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title : str
    url : str
    published_at : Optional[str] = None
    snippet : Optional[str] = None
    source : Optional[str] = None


class EvidencePack(BaseModel):
    evidence : list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder : str = Field(..., description = "eg. [[IMAGE_1]]")
    filename : str = Field(..., description="Save under images/,eg. qkv_flow.png")
    alt : str
    caption : str
    prompt : str = Field(...,descripton = 'Prompt to send the image model.')
    size : Literal['1024*1024', '1024*1536', '1536*1024'] = '1024*1024'
    quality : Literal['low', 'medium', 'high'] = 'medium'


class GlobalImagePlan(BaseModel):
    md_with_placeholders : str
    images : list[ImageSpec] = Field(default_factory=list)


#STATE

class State(TypedDict):
    topic: str

    # routing
    needs_research: bool
    mode: str
    queries: list[str]
    evidence: list[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # worker
    sections: Annotated[list[tuple[int, str]], operator.add]

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict]   # fixed: was 'image_spec' (singular) — now matches decide_images / generate_and_place_images

    final: str


#ROUTER NODE

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=False):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=True):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=True):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=True:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state:State) -> dict:
    
    topic = state['topic']
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content = ROUTER_SYSTEM),
            HumanMessage(content=f"Topic : {topic}")
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state:State) -> str:
    return 'research' if state['needs_research'] else 'orchestrator'


#RESEARCH NODE

def _tavily_search(query: str, max_results: int = 3) -> list[dict]:
    tool = TavilySearchResults(max_results=max_results)   # fixed: was 'results'
    results = tool.invoke({'query': query})

    normalized: list[dict] = []
    for r in results or []:
        normalized.append(
            {
                'title': r.get('title') or "",
                'url': r.get('url') or "",
                'snippet': r.get('snippet') or r.get('content') or "",
                'published_at': r.get('published_at') or r.get('published_date'),  # fixed typo
                'source': r.get('source')
            }
        )

    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state : State) -> dict:

    #take firt 10 queries from state
    queries =(state.get('queries', []) or [])
    max_results = 3

    raw_results : list[dict] =[]

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return{'evidence' : []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
      [
          SystemMessage(content=RESEARCH_SYSTEM),
          HumanMessage(content=f"Raw results : \n{raw_results}")
      ]
    )

    #deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
          dedup[e.url] = e

    return {'evidence' : list(dedup.values())}


#ORCHESTRATOR NODE

def orchestrator(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "")
    planner = llm.with_structured_output(Plan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5-7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3-5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120-450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem -> intuition -> approach -> implementation -> "
                    "trade-offs -> testing/observability -> conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic : {state['topic']}\n"
                    f"Mode : {state.get('mode', '')}\n\n"
                    f"Evidence (Only use for fresh claims, may be empty)\n"
                    f"{[e.model_dump() for e in evidence[:16]]}"
                )
            )
        ]
    )
    
    return {'plan' : plan}


#FANOUT

def fanout(state : State):
    return[
        Send(
            "worker",
            {'task' : task,
             'topic' : state['topic'],
             'mode' : state.get('mode', ''),
             'plan' : state.get('plan'),
             'evidence' : [e.model_dump() for e in state.get('evidence', [])],
            }
        )
        for task in state['plan'].tasks
    ]


#WORKER NODE

import time


def worker(payload: dict) -> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    plan = payload["plan"]
    mode = payload.get("mode", "closed_book")

    bullet_text = "\n-" + "\n-".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"-{e.title} | {e.url} | {getattr(e, 'published_at', 'date : unknown')}".strip()
            for e in evidence[:20]
        )

    # ⏱️ Gemini API Rate-Limit Delay (Har section call ke beech 2 second ka gap)
    time.sleep(7)

    start_time = time.time()

    section_md = (
        llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n"
                        "Hard constraints:\n"
                        "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                        "- Stay close to the Target words (±15%).\n"
                        "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                        "Technical quality bar:\n"
                        "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                        "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                        "- When relevant, include at least one of:\n"
                        "  * a small code snippet (minimal, correct, and idiomatic)\n"
                        "  * a tiny example input/output\n"
                        "  * a checklist of steps\n"
                        "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                        "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                        "- Call out edge cases / failure modes and what to do about them.\n"
                        "- If you mention a best practice, add the 'why' in one sentence.\n"
                        "Markdown style:\n"
                        "- Start with a '## <Section Title>' heading.\n"
                        "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                        "- Avoid fluff. Avoid marketing language.\n"
                        "- If you include code, keep it focused on the bullet being addressed.\n"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Blog : {plan.blog_title}\n"
                        f"Audience : {plan.audience}\n"
                        f"tone : {plan.tone}\n"
                        f"topic : {topic}\n"
                        f"mode : {mode}\n"
                        f"Section : {task.title}\n"
                        f"Goal : {task.goal}\n"
                        f"Target_words : {task.target_words}\n"
                        f"Tags : {task.tags}\n"
                        f"requires_research : {task.requires_research}\n"
                        f"requires_citations : {task.requires_citations}\n"
                        f"requires_code : {task.requires_code}\n"
                        f"Bullets : {bullet_text}\n"
                        f"Evidence (Only use these URLs when citing) : \n{evidence_text}\n"
                    )
                ),
            ]
        )
        .content.strip()
    )

    elapsed_time = round(time.time() - start_time, 2)
    print(f" Section '{task.title}' completed in {elapsed_time}s")

    return {"sections": [(task.id, section_md)]}


#REDUCER

#merge_content -> decide_images -> generate_and_place_images

def merge_content(state:State) -> dict:
    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state['sections'], key=lambda x:x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md" : merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state['merged_md']
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),   # fixed: was string 'DECIDE_IMAGES_SYSTEM'
            HumanMessage(
                content=(
                    f"Topic : {state['topic']}\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            )
        ]
    )

    return {
        'md_with_placeholders': image_plan.md_with_placeholders,
        'image_specs': [img.model_dump() for img in image_plan.images]
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:   # renamed: was _gemini_generate_images_bytes
    """
    Return raw image bytes generated by gemini.
    requires : pip install google-genai
    env var : GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY is not set.')

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ]
        )
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])   # now matches renamed function
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n\n"
                    f"> **Alt:** {spec.get('alt', '')}\n\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


# BUILD REDUCER SUBGRAPH

reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

reducer_subgraph


#BUILD MAIN GRAPH

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app