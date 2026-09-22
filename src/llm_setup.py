from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_anthropic import ChatAnthropic

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
together_api = os.getenv("TOGETHER_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")
xai_api = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
anthropic_api = os.getenv("ANTHROPIC_API_KEY")

LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "gpt-5.2-2025-12-11")
# PDF rendering quality (DPI) - affects all extractions
# Higher DPI materially improves table readability in vision extraction.
LLM_PDF_RENDER_DPI = int(os.getenv("LLM_PDF_RENDER_DPI", "300"))

# Provider selection for extraction: "openai", "anthropic", or "xai".
# If set to "auto", infer from the extraction model name.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

def _infer_provider_from_model(model: str) -> str:
    m = (model or "").strip().lower()
    if not m:
        return "openai"
    if m.startswith("claude-") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("grok-") or m.startswith("xai-"):
        return "xai"
    if m.startswith("gpt-") or m.startswith("o"):
        return "openai"
    return "openai"

if LLM_PROVIDER in {"", "auto"}:
    LLM_PROVIDER = _infer_provider_from_model(LLM_EXTRACTION_MODEL)

# Set environment variables for provider code to read
os.environ["LLM_PROVIDER"] = LLM_PROVIDER
os.environ["LLM_EXTRACTION_MODEL"] = LLM_EXTRACTION_MODEL
os.environ["LLM_PDF_RENDER_DPI"] = str(LLM_PDF_RENDER_DPI)

client = OpenAI(api_key=openai_api) if openai_api else None


def _require_api_key(provider: str) -> None:
    p = (provider or "").strip().lower()
    if p == "openai" and not openai_api:
        raise ValueError(
            "OPENAI_API_KEY is not set, but GRADING_PROVIDER is 'openai'. "
            "Set OPENAI_API_KEY or switch GRADING_PROVIDER/LLM_GRADER_MODEL."
        )
    if p in {"anthropic", "claude"} and not anthropic_api:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set, but GRADING_PROVIDER is 'anthropic'. "
            "Set ANTHROPIC_API_KEY or switch GRADING_PROVIDER/LLM_GRADER_MODEL."
        )
    if p in {"xai", "grok"} and not xai_api:
        raise ValueError(
            "XAI_API_KEY (or GROK_API_KEY) is not set, but GRADING_PROVIDER is 'xai'. "
            "Set XAI_API_KEY or switch GRADING_PROVIDER/LLM_GRADER_MODEL."
        )

def _parse_model_spec(model_spec: str, default_provider: str) -> tuple[str, str]:
    """Parse 'provider:model' or plain 'model' into (provider, model)."""
    spec = (model_spec or "").strip()
    if ":" in spec:
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        return provider, model

    provider = default_provider
    if provider in {"", "auto"}:
        provider = _infer_provider_from_model(spec)
    return provider, spec


# Grading provider/model can be changed from env without code edits.
# Supported providers: openai, anthropic, xai
GRADING_PROVIDER = os.getenv("GRADING_PROVIDER", "auto").strip().lower()
LLM_GRADER_MODEL = os.getenv("LLM_GRADER_MODEL", "gpt-5.2-2025-12-11")
LLM_CHAIN_MODEL = os.getenv("LLM_CHAIN_MODEL", LLM_GRADER_MODEL)

GRADING_PROVIDER, LLM_GRADER_MODEL = _parse_model_spec(LLM_GRADER_MODEL, GRADING_PROVIDER)
chain_provider, LLM_CHAIN_MODEL = _parse_model_spec(LLM_CHAIN_MODEL, GRADING_PROVIDER)
if chain_provider != GRADING_PROVIDER:
    # Keep it simple: one provider for the workspace.
    GRADING_PROVIDER = chain_provider


def _build_chat_model(provider: str, model: str, temperature: float = 0):
    _require_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=openai_api,  # type: ignore[arg-type]
            temperature=temperature,
        )
    if provider in ["anthropic", "claude"]:
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,  # type: ignore[call-arg]
            api_key=anthropic_api,  # type: ignore[arg-type]
            max_tokens=16000,  # type: ignore[call-arg]  # Anthropic default is ~1024; grading 82 criteria needs far more
            max_retries=2,
        )
    if provider in ["xai", "grok"]:
        return ChatXAI(  # type: ignore[call-arg]
            model=model,
            xai_api_key=xai_api,  # type: ignore[arg-type]
            temperature=temperature,
            max_retries=2,
        )
    raise ValueError(
        f"Unsupported GRADING_PROVIDER '{provider}'. "
        "Supported: openai, anthropic, xai"
    )


llm_grader = _build_chat_model(GRADING_PROVIDER, LLM_GRADER_MODEL, temperature=0)
llm = _build_chat_model(GRADING_PROVIDER, LLM_CHAIN_MODEL, temperature=0)
