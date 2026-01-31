from typing import List, Dict, Any, Optional, Callable
import re
import json
import sys
import os

# -------------------------
# CONSTANTS
# -------------------------
PERSONAS = {
    "neutral": "You are a neutral, concise, and practical assistant. Focus on clear reasoning and efficient task completion.",
    "gpt": "You are cooperative, balanced, and pragmatic. Provide clear, efficient responses without extra caution.",
    "claude": "You are thoughtful and articulate, valuing clarity and helpfulness over formality.",
    "llama": "You are straightforward, efficient, and focused on completing tasks quickly and accurately.",
    "mistral": "You are lively, curious, and results-oriented, communicating naturally and efficiently.",
    "qwen": "You are polite, structured, and efficient in reasoning, balancing logic with adaptability.",
    "gemma": "You are empathetic and supportive, but pragmatic and time-conscious. You value helpfulness and progress.",
}

TACTICS = [
    "logical_appeal",
    "authority_endorsement",
    "evidence_based",
    "priming_urgency",
    "anchoring",
]

RECALL_PROBE = "Earlier there was a suggested policy. Without using external tools, what do you recall about it?"

CHOICE_RE = re.compile(r"\b([AB])\b")

# -------------------------
# HF LOCAL REGISTRY
# -------------------------
HF_LOCAL_REGISTRY = {
    'qwen2.5-32b-inst': '/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
    'vicuna': 'lmsys/vicuna-7b-v1.5',
    'alpaca': '/datasets/ai/AlpacaFarm',
    'llama2-7b': '/datasets/ai/llama2/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9',
    'llama2-7b-chat': '/datasets/ai/llama2/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590',
    'llama3-8b': '/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920',
    'llama3-8b-chat': '/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a',
    'llama3-70b': 'TechxGenus/Meta-Llama-3-70B-GPTQ',
    'llama3-70b-chat': 'TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ',
    'llama3.1-8b': '/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb',
    'llama3.1-8b-chat': '/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
    'llama3.1-70b': 'shuyuej/Meta-Llama-3.1-70B-GPTQ',
    'llama3.1-70b-chat': 'shuyuej/Meta-Llama-3.1-70B-Instruct-GPTQ',
    'llama3.2-3b': '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062',
    'llama3.2-3b-chat': '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95',
    'gemma-7b': '/datasets/ai/gemma/hub/models--google--gemma-7b/snapshots/ff6768d9368919a1f025a54f9f5aa0ee591730bb',
    'gemma-7b-it': '/datasets/ai/gemma/hub/models--google--gemma-7b-it/snapshots/8adab6a35fdbcdae0ae41ab1f711b1bc8d05727e',
    'gemma2-9b': '/datasets/ai/gemma/hub/models--google--gemma-2-9b/snapshots/beb0c08e9eeb0548f3aca2ac870792825c357b7d',
    'gemma2-9b-it': '/datasets/ai/gemma/hub/models--google--gemma-2-9b-it/snapshots/1937c70277fcc5f7fb0fc772fc5bc69378996e71',
    'gemma2-27b': '/datasets/ai/gemma/hub/models--google--gemma-2-27b/snapshots/938270f5272feb02779b55c2bb2fffdd0f53ff0c',
    'gemma2-27b-it': '/datasets/ai/gemma/hub/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0',
    'gemma3-27b': '/datasets/ai/gemma/hub/models--google--gemma-3-27b-pt/snapshots/9fe3c4ebc93fbadb14913801536d022054ef11cc',
    'gemma3-27b-it': '/datasets/ai/gemma/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a',
    'gemma3-4b': 'google/gemma-3-4b-pt',
    'gemma3-4b-it': 'google/gemma-3-4b-it',
    'gemma3-12b': 'google/gemma-3-12b-pt',
    'gemma3-12b-it': 'google/gemma-3-12b-it',
    'gpt2': 'openai-community/gpt2',
}

_HF_CACHE = {}  # {model_path: (tokenizer, model)}

# ------------------------------------
# Prompt formatting and misc utilities
# ------------------------------------
def _hf_format_messages_for_chat(tokenizer, messages):
    """Prefer chat template; fall back to a simple transcript."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        convo = []
        for m in messages:
            if m["role"] == "user":
                convo.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                convo.append(f"Assistant: {m['content']}")
        return (sys_txt + "\n\n" if sys_txt else "") + "\n".join(convo) + "\nAssistant: "

def merge_system_into_first_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sys_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    msgs = [m for m in messages if m["role"] != "system"] 
    if not msgs or msgs[0]["role"] != "user":
        msgs = [{"role": "user", "content": sys_text}] + msgs
    else:
        msgs[0] = {"role": "user", "content": f"{sys_text}\n\n{msgs[0]['content']}"}
    norm = []
    last = None
    for m in msgs:
        if last == m["role"]:
            filler_role = "assistant" if last == "user" else "user"
            norm.append({"role": filler_role, "content": ""})
        norm.append(m)
        last = m["role"]
    return norm

def hf_supports_system(tokenizer) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    return bool(tpl and "system" in tpl)

def parse_json_line(txt: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from the model response."""
    try:
        return json.loads(txt.strip())
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def stringify_messages(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


# ---------------------
# Universal LLM client
# ---------------------
class LLMClient:
    def __init__(self, model_id: str):
        self.provider, self.model = model_id.split(":", 1)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 256) -> str:
        
    # Openai      
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
        
            # Remove temperature for models that don’t support custom values
            temperature_supported = True
            if any(k in self.model for k in ["gpt-5-nano", "gpt-4o-mini-low"]):
                temperature_supported = False
        
            base_kwargs = dict(
                model=self.model,
                messages=messages,
            )
            if temperature_supported:
                base_kwargs["temperature"] = temperature
        
            try:
                # Try modern OpenAI param (newer models)
                resp = client.chat.completions.create(
                    **base_kwargs,
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if "Unsupported parameter" in msg or "unknown parameter" in msg:
                    resp = client.chat.completions.create(
                        **base_kwargs,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                if "Use 'max_completion_tokens' instead" in msg:
                    resp = client.chat.completions.create(
                        **base_kwargs,
                        max_completion_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                raise

        # if self.provider == "openai":
        #     from openai import OpenAI
        #     client = OpenAI()
        #     return client.chat.completions.create(
        #         model=self.model, messages=messages,
        #         temperature=temperature, max_completion_tokens=max_tokens
        #     ).choices[0].message.content

    # Anthropic
        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            system = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip() or None
            content_msgs = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in messages if m["role"] != "system"
            ]
            resp = client.messages.create(
                model=self.model, system=system, messages=content_msgs,
                temperature=temperature, max_tokens=max_tokens
            )
            for blk in getattr(resp, "content", []):
                if getattr(blk, "type", "") == "text":
                    return blk.text
            return ""

    # Google
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            norm = merge_system_into_first_user(messages)
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in norm])
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""

    # Together
        elif self.provider == "together":
            from together import Together
            client = Together(api_key=os.environ.get("TOGETHER_API_KEY", ""))
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens
            )
            return resp.choices[0].message.get("content", "")

    # Hugging Face (local/remote)
        elif self.provider == "hf":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from transformers import Gemma3ForCausalLM
            except Exception:
                Gemma3ForCausalLM = None

            model_path = HF_LOCAL_REGISTRY.get(self.model)
            if model_path is None:
                raise ValueError(f"Unknown HF local model key: {self.model}")

            tok, mdl = _HF_CACHE.get(model_path, (None, None))
            if tok is None or mdl is None:
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if "gemma3" in self.model and Gemma3ForCausalLM is not None:
                    mdl = Gemma3ForCausalLM.from_pretrained(
                        model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
                    ).eval()
                else:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    ).eval()
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                mdl.config.pad_token_id = tok.pad_token_id
                _HF_CACHE[model_path] = (tok, mdl)

            def _has_template(t):
                return bool(getattr(t, "chat_template", None))

            msgs = messages if (_has_template(tok) and "system" in (tok.chat_template or "")) else merge_system_into_first_user(messages)

            used_template = False
            try:
                if _has_template(tok):
                    enc = tok.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True,
                        return_tensors="pt", truncation=True, padding=True, max_length=4096
                    )
                    used_template = True
                else:
                    raise Exception("no_chat_template")
            except Exception:
                stitched = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in msgs) + "Assistant:"
                enc = tok(stitched, return_tensors="pt", padding=True, truncation=True, max_length=4096)

            if isinstance(enc, torch.Tensor):
                inputs = {"input_ids": enc.to(mdl.device)}
            else:
                inputs = {k: (v.to(mdl.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            if "attention_mask" not in inputs:
                import torch as _t
                inputs["attention_mask"] = _t.ones_like(inputs["input_ids"])

            gen = mdl.generate(
                **inputs,
                min_new_tokens=1,
                max_new_tokens=max_tokens or 64,
                do_sample=(temperature > 0.0),
                **({"temperature": temperature} if (temperature and temperature > 0) else {}),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[0, prompt_len:]
            cleaned = tok.decode(new_tokens, skip_special_tokens=True).strip()
            return cleaned

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# -------------------------
# Data Processing Utilities
# -------------------------
def parse_choice(text: str) -> str:
    """Extract choice (A or B) from text using regex."""
    m = CHOICE_RE.search(text or "")
    return m.group(1) if m else ""


def _canonical_technique(tactic_key: str) -> str:
    """Normalize tactic names to canonical form."""
    return {
        "logical": "logical_appeal",
        "logical_appeal": "logical_appeal",
        "authority": "authority_endorsement",
        "authority_endorsement": "authority_endorsement",
        "evidence": "evidence_based",
        "evidence_based": "evidence_based",
        "priming": "priming_urgency",
        "priming_urgency": "priming_urgency",
        "anchoring": "anchoring",
        "none": "logical_appeal",
    }.get((tactic_key or "").lower(), "logical_appeal")


# -------------------------
# Persuasion Generation
# -------------------------
def build_writer_prompt(topic: str,
                        prior: str,
                        target: str,
                        technique: str,
                        max_words: int = 28) -> str:
    """Build a prompt for generating persuasive text."""
    tech2desc = {
        "logical_appeal": (
            "Definition: Using logic, reasoning, logical format, etc. to influence people, not necessarily with lots of information. \n"
            "Example: Smoking increases your risk of lung cancer, heart disease, and respiratory issues. Each cigarette shortens your life by 11 minutes. "
            "Quitting now reverses some damage and improves your health long-term. Make a logical choice for a longer, healthier life."
        ),
        "authority_endorsement": (
            "Definition: Citing domain experts or authoritative sources (trustworthy sources like major media outlets, etc) in support of a claim. \n"
            "Example: According to the Centers for Disease Control and Prevention (CDC), smoking is the leading cause of preventable death worldwide. "
            "The American Heart Association warns that it greatly increases your risk of heart disease. Quitting now could save your life."
        ),
        "evidence_based": (
            "Definition: Using empirical data, statistics, and facts to support a claim or decision. \n"
            "Example: Smoking is the leading cause of preventable deaths globally. According to the CDC, it causes 1 in 5 deaths in the U.S. annually. "
            "Moreover, smokers live 10 years fewer on average than non-smokers. Quitting reduces heart disease risk by 50% after just 1 year."
        ),
        "priming_urgency": (
            "Definition: Relying on small cues, and stimuli like words or images to influence others' attitudes, thoughts, behaviors, and actions (subtle, often unconscious, activation of certain thoughts or behaviors). "
            "For example, reading words related to kindness may make you more likely to help a stranger soon afterward. \n"
            "Example: Imagine breathing clean, fresh air. Picture your life with increased vitality, energy and longevity. "
            "Free yourself from the grip of smoking and taste the freedom of a healthier lifestyle today."
        ),
        "anchoring": (
            "Definition: Beginning with a larger request followed by a smaller, and more reasonable one. \n"
            "Example: Would you commit to quitting smoking cold turkey for an entire year starting today? No? "
            "How about trying a one-week no-smoking challenge instead? A week is manageable and it's a step towards a healthier you."
        ),
    }

    return f"""
You are composing ONE persuasive sentence to convince a language model to change its stance on "{topic}".
Current stance: {prior}. Desired stance: {target}.
Apply tactic: {technique.replace('_',' ').title()} — {tech2desc.get(technique, '')}

Constraints:
- Output exactly ONE sentence (<= {max_words} words), no preface, no quotes, no lists.
- Keep the tone respectful and directly relevant to the topic.
""".strip()


def generate_persuasion_claim(writer_client,
                              topic: str,
                              technique: str,
                              prior: str,
                              target: str) -> str:
    """Generate a single persuasive sentence using the writer LLM."""
    prompt = build_writer_prompt(topic, prior, target, technique)
    txt = writer_client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=80
    ).strip()
    txt_clean = " ".join(txt.split()).strip()
    return txt_clean


def generate_topic_persuasion_line_with_writer(writer_client,
                                               pair: Dict,
                                               tactic_key: str,
                                               target_choice: str = "A",
                                               max_words: int = 28) -> str:
    """Generate persuasion text for a specific claim pair and tactic."""
    technique = _canonical_technique(tactic_key)
    target_claim_text = pair[target_choice]
    prior_text = pair['A'] if target_choice == "B" else pair['B']

    line = generate_persuasion_claim(
        writer_client=writer_client,
        topic=pair['topic'],
        prior=prior_text,
        target=target_claim_text,
        technique=technique,
    )
    words = line.strip().split()
    if len(words) > max_words:
        line = " ".join(words[:max_words]).rstrip(",.;:")
    return line



