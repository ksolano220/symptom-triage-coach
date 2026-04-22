"""JSON schema for the pre-visit prep output.

Defines the exact shape of the structured response the fine-tuned model
must produce. Used for:
  - validating teacher-model outputs during synthetic data generation
  - scoring schema-adherence during evaluation
  - rendering the Gradio UI at inference time
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "systems",
        "possible_causes",
        "red_flags",
        "questions_to_prepare_for",
    ],
    "properties": {
        "systems": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "Body systems that might be involved.",
        },
        "possible_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "likelihood", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "likelihood": {
                        "type": "string",
                        "enum": ["common", "less_common", "rare", "serious"],
                    },
                    "description": {"type": "string"},
                },
            },
            "minItems": 2,
            "maxItems": 8,
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
            "description": "Symptoms that warrant immediate medical attention.",
        },
        "questions_to_prepare_for": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
            "description": "Questions the doctor will likely ask the patient.",
        },
    },
    "additionalProperties": False,
}


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
