"""Hugging Face Space entry point for Symptom Triage Coach.

Self-contained Gradio app. Loads Qwen2.5-1.5B-Instruct + the LoRA adapter
from the Hub, generates schema-valid JSON from a patient's symptom
description, and renders it as formatted markdown.

Supports Spanish-language input: when the user selects Español, the
symptom description is translated to English before being sent to the
model. The model itself is English-only, so output stays in English.
"""

import json

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "ksolano220/symptom-triage-coach"

SYSTEM_PROMPT = """You are a pre-visit medical prep assistant. Your job is to help patients prepare for a doctor's visit by explaining what the visit might involve given their symptoms.

You will receive a patient's plain-language symptom description. Respond with a JSON object matching this schema exactly:

{
  "systems": [string],              // body systems that might be involved, lowercase
  "possible_causes": [
    {
      "name": string,               // condition name, in plain English when possible
      "likelihood": "common" | "less_common" | "rare" | "serious",
      "description": string         // one sentence explanation a patient can understand
    }
  ],
  "red_flags": [string],            // warning signs requiring immediate care
  "questions_to_prepare_for": [string]  // questions the doctor will likely ask
}

CRITICAL RULES:
- You are NOT diagnosing. You are helping the patient prepare.
- Never claim a specific condition. Use "might be" or "could include" framing.
- Always include red flags for any symptom that could indicate emergency.
- Keep descriptions under 20 words each, plain language.
- Output ONLY valid JSON. No preamble, no explanation outside the JSON."""

EXAMPLES = [
    "I have chest pain when I breathe deeply",
    "My head has been pounding for hours",
    "I get dizzy when I stand up quickly",
    "My stomach hurts in the upper right",
    "I've been short of breath walking up stairs",
]

DESCRIPTION = """
# Symptom Triage Coach

Helps you prepare for a doctor's visit. Describe a symptom and this tool suggests body systems that might be involved, possible causes, red flags to watch for, and questions the doctor will likely ask.

**Not medical advice.** This is a pre-visit prep tool for research and portfolio purposes, not a diagnostic product. For any real medical concern, see a licensed clinician.

First request takes ~60 seconds while the model loads. Subsequent requests are ~15 seconds on free-tier CPU.
"""

LIKELIHOOD_BADGE = {
    "common": "common",
    "less_common": "less common",
    "rare": "rare",
    "serious": "serious ⚠",
}

print("Loading base model and adapter (this takes ~60s)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()
print("Model ready.")


def generate_json(text: str) -> dict | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def format_markdown(data: dict) -> str:
    parts = []

    systems = data.get("systems", [])
    if systems:
        parts.append("### Body systems to consider")
        parts.append(", ".join(s.capitalize() for s in systems))
        parts.append("")

    causes = data.get("possible_causes", [])
    if causes:
        parts.append("### Possible causes")
        for c in causes:
            name = c.get("name", "")
            likelihood = LIKELIHOOD_BADGE.get(c.get("likelihood", ""), c.get("likelihood", ""))
            desc = c.get("description", "")
            parts.append(f"- **{name}** _(likelihood: {likelihood})_ — {desc}")
        parts.append("")

    red_flags = data.get("red_flags", [])
    if red_flags:
        parts.append("### Red flags — seek immediate care if you notice")
        for flag in red_flags:
            parts.append(f"- {flag}")
        parts.append("")

    questions = data.get("questions_to_prepare_for", [])
    if questions:
        parts.append("### Questions the doctor will likely ask")
        for q in questions:
            parts.append(f"- {q}")

    return "\n".join(parts)


def translate_to_english(text: str) -> str:
    """Translate Spanish input to English. Falls back to the raw input on failure."""
    if GoogleTranslator is None:
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except Exception:
        return text


def summarize(text: str, language: str):
    text = (text or "").strip()
    if not text:
        return "", ""
    if language == "Español":
        text_en = translate_to_english(text)
    else:
        text_en = text
    data = generate_json(text_en)
    if data is None:
        return "_Model output was not valid JSON. Try rephrasing the symptom._", ""
    markdown = format_markdown(data)
    raw = json.dumps(data, indent=2)
    return markdown, raw


CUSTOM_CSS = """
#symptom-input textarea,
#symptom-input input,
#symptom-input .scroll-hide,
.gradio-container textarea,
.gradio-container .input-textarea textarea {
    font-size: 20px !important;
    line-height: 1.6 !important;
    padding: 16px !important;
}
#symptom-input label,
#symptom-input .label-wrap {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Language toggle styled as prominent pill switch */
#language-toggle {
    margin-top: 8px !important;
    padding: 12px 14px !important;
    background: #f5f7fb !important;
    border: 1px solid #e3e6ef !important;
    border-radius: 10px !important;
}
#language-toggle .label-wrap,
#language-toggle > label,
#language-toggle span[data-testid="block-label"] {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
#language-toggle label {
    font-size: 15px !important;
    padding: 6px 14px !important;
    border-radius: 999px !important;
    cursor: pointer !important;
}
#language-toggle input[type="radio"]:checked + span,
#language-toggle label:has(input:checked) {
    background: #121631 !important;
    color: #ffffff !important;
}
"""


def _update_input_ui(language: str):
    if language == "Español":
        return gr.update(
            label="Describa su síntoma",
            placeholder="ej. Tengo dolor en el pecho al respirar profundamente",
        )
    return gr.update(
        label="Describe your symptom",
        placeholder="e.g. I have chest pain when I breathe deeply",
    )


with gr.Blocks(title="Symptom Triage Coach", css=CUSTOM_CSS) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            source = gr.Textbox(
                label="Describe your symptom",
                lines=4,
                placeholder="e.g. I have chest pain when I breathe deeply",
                elem_id="symptom-input",
            )
            language = gr.Radio(
                choices=["English", "Español"],
                value="English",
                label="Type in your language / Escribe en tu idioma",
                elem_id="language-toggle",
            )
            run_btn = gr.Button("Prep for doctor visit", variant="primary")
        with gr.Column():
            output_md = gr.Markdown(label="Pre-visit prep")

    with gr.Accordion("Raw JSON output", open=False):
        output_raw = gr.Code(language="json", label=None)

    gr.Examples(examples=EXAMPLES, inputs=source)
    language.change(_update_input_ui, inputs=language, outputs=source)
    run_btn.click(summarize, inputs=[source, language], outputs=[output_md, output_raw])


if __name__ == "__main__":
    demo.launch()
