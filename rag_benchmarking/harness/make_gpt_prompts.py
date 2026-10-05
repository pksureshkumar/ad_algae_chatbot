"""Generate copy-paste prompts for the manually collected GPT arm.

The web interface has no system-role field, so NEUTRAL_SYSTEM is folded into the
user message. Everything else is byte-identical to what the local no_retrieval
arm received, so the only difference between those arms is the model.

Outputs:
    gpt_arm/prompts.md    one ready-to-paste block per question
    gpt_arm/answers.md    template to paste responses back into
    gpt_arm/parse_answers.py  converts answers.md -> runs/gpt_strict.jsonl

Usage:  python make_gpt_prompts.py
"""
import argparse
import json

from paths import DATASET, BENCH
from retrieval import NEUTRAL_SYSTEM, NO_CONTEXT_PROMPT

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="gpt_strict_arm",
                 help="directory under rag_benchmarking/ to write into")
_ap.add_argument("--note", default="",
                 help="line added to the answers header, e.g. 'SEARCH MUST BE OFF'")
_ap.add_argument("--no-tools", action="store_true",
                 help="prepend an explicit instruction not to search or use tools")
_args = _ap.parse_args()

# Astra invokes browsing on factual queries even with web search disabled in the
# account settings, so the UI toggle alone does not produce an ungrounded arm.
# This is a prompt-level attempt at the same thing. Deliberately says "training
# data" rather than "knowledge base" or "corpus": telling a model it has a corpus
# it does not have is what made the first no_retrieval run invent citations.
NO_TOOLS_PREFIX = (
    "Do not use any tools. Do not search the web, browse, or retrieve documents. "
    "Answer only from knowledge already in your training data. If you cannot "
    "recall a specific value, say so rather than looking it up or estimating.\n\n"
)

OUT = BENCH / _args.out
OUT.mkdir(exist_ok=True)

rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]

RULES = """# GPT arm — manual collection

**{n} prompts. One per FRESH conversation.** Do not put more than one question in
a conversation, and do not reuse a conversation between questions.

Before starting, check three settings:

1. **Web search / browsing OFF.** With search on this is a web-retrieval system,
   not an ungrounded baseline, and the comparison measures something else.
2. **Memory / personalisation OFF**, or use temporary chats. Memory carries
   context between conversations and reintroduces contamination sideways.
3. Record the **model name shown in the interface** and **today's date**. Both go
   in the methods, since this arm has no pinned snapshot.

Paste each block below as the first and only message of a new chat. Copy the full
reply into `answers.md` under the matching heading.

---

"""


def main():
    blocks = [RULES.format(n=len(rows))]
    for r in rows:
        prompt = NEUTRAL_SYSTEM + "\n\n"
        if _args.no_tools:
            prompt += NO_TOOLS_PREFIX
        prompt += NO_CONTEXT_PROMPT.format(question=r["user_input"])
        blocks.append(
            f"## {r['id']}\n\n"
            f"<!-- source: {r['source_filename']} | stratum: {r['stratum']} | "
            f"gold: {', '.join(r['gold_values'])} -->\n\n"
            "```\n" + prompt.strip() + "\n```\n\n---\n"
        )
    (OUT / "prompts.md").write_text("\n".join(blocks), encoding="utf-8")

    # Preserve anything already pasted: regenerating must never destroy
    # collected answers, which cannot be reproduced without redoing the chats.
    answers_path = OUT / "answers.md"
    existing = {}
    if answers_path.exists():
        import re
        text = answers_path.read_text(encoding="utf-8")
        for qid, body in re.findall(r"^## (Q\d+)\b.*?\n+```(.*?)```", text, re.S | re.M):
            if body.strip():
                existing[qid] = body.strip()
        if existing:
            print(f"  preserving {len(existing)} already-pasted answer(s): "
                  f"{', '.join(sorted(existing))}")

    tmpl = ["# GPT arm — collected answers", ""]
    if _args.note:
        tmpl += [f"> **{_args.note}**", ""]
    tmpl += ["Model shown in interface: __________    Date collected: __________",
            "Reasoning setting: __________",
            "Search off: [ ]    Memory off / temporary chat: [ ]", "",
            "Paste each full reply inside the fences under its question.",
            "The question is repeated here so you can confirm you are pasting the",
            "reply to the right one — mis-mapping is the main risk when running",
            "several chats at once.", "", "---", ""]
    for r in rows:
        tmpl += [f"## {r['id']}", "",
                 f"> {r['user_input']}", "",
                 f"*stratum: {r['stratum']} · source: {r['source_filename']}*", "",
                 "```", existing.get(r["id"], ""), "```", "", "---", ""]
    answers_path.write_text("\n".join(tmpl), encoding="utf-8")

    # Copy the maintained parser rather than regenerating it from an embedded
    # string: the embedded copy went stale and silently overwrote a fixed one.
    import shutil
    canonical = BENCH / 'gpt_websearch_arm' / 'parse_answers.py'
    if canonical.exists() and canonical.parent != OUT:
        shutil.copy(canonical, OUT / 'parse_answers.py')
        print(f"copied parser from {canonical.parent.name}/")

    print(f"wrote {OUT / 'prompts.md'}  ({len(rows)} prompts)")
    print(f"wrote {OUT / 'answers.md'}")
    print(f"wrote {OUT / 'parse_answers.py'}")


if __name__ == "__main__":
    main()
