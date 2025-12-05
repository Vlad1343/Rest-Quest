from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency at runtime
    genai = None

BASE_PATH = Path(__file__).parent

# Load environment variables from both the repository root and analysis/.env if present.
load_dotenv(BASE_PATH.parent / ".env")
load_dotenv(BASE_PATH / ".env")

SYSTEM_PROMPT_PATH = BASE_PATH / "gemini_system_prompt.md"
TRAVEL_DATA_PATH = BASE_PATH / "mock_travel_data.json"
MOCK_REQUEST_PATH = BASE_PATH / "mock_gemini_request.json"

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
TRAVEL_DATA = json.loads(TRAVEL_DATA_PATH.read_text(encoding="utf-8"))

_client: Optional["genai.Client"] = None

HIGH_STRESS = ["overwhelmed", "stressed", "anxious", "pressure", "deadline", "burnout"]
LOW_STRESS = ["calm", "peaceful", "relaxed", "rested", "content"]
HIGH_ENERGY = ["energetic", "active", "excited", "motivated", "ready"]
LOW_ENERGY = ["tired", "exhausted", "drained", "fatigued", "sleepy"]
TRIGGER_KEYWORDS = ["work", "deadline", "deadlines", "relationships", "family", "travel", "money", "health"]
CHAT_STYLE_PROMPT = """You are Serenity, a mindful travel concierge. Your job is to respond like a calm guide:
- acknowledge feelings with grounded language
- keep replies between 2-4 sentences
- weave in subtle sensory imagery and somatic cues
- invite gentle next steps rather than commands
Avoid bullet lists. Keep tone empathetic, not clinical."""


@dataclass
class AnalysisResult:
    stress: int
    energy: int
    valence: int
    dominant_emotion: str
    triggers: List[str]
    recommendations: List[str]
    overall_mood: str
    voice_summary: str
    emotion_summary: str
    destinations: List[Dict[str, Any]]


def load_mock_request() -> Dict[str, Any]:
    return json.loads(MOCK_REQUEST_PATH.read_text(encoding="utf-8"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ensure_client() -> Optional["genai.Client"]:
    global _client
    if _client or genai is None:
        return _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(payload: Dict[str, Any]) -> str:
    request_block = json.dumps(payload, indent=2)
    travel_block = json.dumps(TRAVEL_DATA, indent=2)
    return f"{SYSTEM_PROMPT}\n\nRequest:\n{request_block}\n\nTravel Data:\n{travel_block}"


def _parse_response(text: str | None) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip potential markdown fences such as ```json ... ```
        parts = cleaned.split("```")
        cleaned = "".join(parts[1:-1]).strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _score_from_keywords(text: str, words: Iterable[str], delta: int) -> int:
    score = 0
    for word in words:
        if word in text:
            score += delta
    return score


def _compute_scores(text: str) -> Tuple[int, int, int]:
    lower = text.lower()
    stress = 38 + _score_from_keywords(lower, HIGH_STRESS, 12) - _score_from_keywords(lower, LOW_STRESS, 10)
    energy = 42 + _score_from_keywords(lower, HIGH_ENERGY, 10) - _score_from_keywords(lower, LOW_ENERGY, 12)
    valence = 60 - (stress - 55) * 0.5 + (energy - 50) * 0.4
    return (
        int(_clamp(stress, 12, 95)),
        int(_clamp(energy, 8, 95)),
        int(_clamp(valence, 5, 95)),
    )


def _aggregate_emotions(visual_entries: Iterable[Dict[str, Any]]) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    for entry in visual_entries:
        spectrum = entry.get("spectrum") or {}
        for emotion, score in spectrum.items():
            totals[emotion] = totals.get(emotion, 0.0) + float(score)
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def _dominant_emotion_from_visual(emotion_ranking: List[Tuple[str, float]], fallback: str) -> str:
    if not emotion_ranking:
        return fallback
    return emotion_ranking[0][0]


def _default_dominant(stress: int, valence: int) -> str:
    if stress > 75:
        return "stressed"
    if stress > 60:
        return "anxious"
    if valence > 65:
        return "happy"
    if valence < 35:
        return "sad"
    return "calm"


def _build_recommendations(stress: int, energy: int, valence: int) -> List[str]:
    recs: List[str] = []
    if stress > 75:
        recs.extend(["deep ocean therapy", "digital sabbath"])
    if energy < 40:
        recs.extend(["restorative breathwork", "guided journaling"])
    if valence < 40:
        recs.extend(["nature immersion", "awe practices"])
    if not recs:
        recs.extend(["maintenance rituals", "gratitude scans"])
    return recs


def _detect_triggers(text: str) -> List[str]:
    lower = text.lower()
    found = [word for word in TRIGGER_KEYWORDS if word in lower]
    return found or ["work rhythm", "self-expectations"]


def _build_destinations(emotion_order: List[str], triggers: List[str]) -> List[Dict[str, Any]]:
    if not emotion_order:
        emotion_order = ["default"]
    emotion_order = emotion_order + ["default"]
    choices: List[Dict[str, Any]] = []
    seen = set()
    for emotion in emotion_order:
        for destination in TRAVEL_DATA.get(emotion, []):
            key = (destination["name"], destination.get("region"))
            if key in seen:
                continue
            reason_trigger = triggers[0] if triggers else "recent stress"
            reason = (
                destination.get("reason")
                or f"Calms {reason_trigger} with {destination.get('vibe', 'restorative rituals')}."
            )
            enriched = {**destination, "reason": reason, "emotion": emotion}
            choices.append(enriched)
            seen.add(key)
            if len(choices) >= 5:
                return choices
    return choices


def _summaries(payload: Dict[str, Any], emotion_ranking: List[Tuple[str, float]], dominant: str) -> Tuple[str, str]:
    voice = payload.get("voiceTranscript") or ""
    voice_summary = "User shared brief preferences."
    if voice.strip():
        snippet = voice.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177].rstrip() + "..."
        voice_summary = snippet

    if emotion_ranking:
        top_pairs = [f"{name} {score:.0%}" for name, score in emotion_ranking[:2]]
        emotion_summary = f"Dominant mood {dominant}. Visual spectra hint at {', '.join(top_pairs)}."
    else:
        emotion_summary = f"Dominant mood inferred as {dominant}."
    return voice_summary, emotion_summary


def _fallback_bundle(payload: Dict[str, Any], entry: Optional[str]) -> AnalysisResult:
    voice_text = payload.get("voiceTranscript") or entry or ""
    stress, energy, valence = _compute_scores(voice_text)
    emotion_ranking = _aggregate_emotions(payload.get("visualEmotionTranscript", []))
    dominant = _dominant_emotion_from_visual(emotion_ranking, _default_dominant(stress, valence))
    triggers = _detect_triggers(voice_text)
    recommendations = _build_recommendations(stress, energy, valence)
    destinations = _build_destinations([item[0] for item in emotion_ranking] or [dominant], triggers)
    voice_summary, emotion_summary = _summaries(payload, emotion_ranking, dominant)
    return AnalysisResult(
        stress=stress,
        energy=energy,
        valence=valence,
        dominant_emotion=dominant,
        triggers=triggers,
        recommendations=recommendations,
        overall_mood=dominant,
        voice_summary=voice_summary,
        emotion_summary=emotion_summary,
        destinations=destinations,
    )


def _call_gemini(payload: Dict[str, Any]) -> Dict[str, Any]:
    client = _ensure_client()
    if client is None:
        raise RuntimeError("Gemini client is not available. Install google-generativeai and set GEMINI_API_KEY.")
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=_build_prompt(payload),
        config={
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            "max_output_tokens": int(os.getenv("GEMINI_MAX_OUTPUT", "1024")),
        },
    )
    parsed = _parse_response(getattr(response, "text", None))
    if not parsed:
        raise RuntimeError("Gemini returned an empty or invalid response.")
    return parsed


def generate_analysis(
    request_payload: Optional[Dict[str, Any]] = None,
    *,
    entry: Optional[str] = None,
    allow_gemini: bool = True,
) -> Dict[str, Any]:
    """Generate an analysis payload either via Gemini or deterministic fallbacks."""

    payload = dict(request_payload or {})
    if entry and not payload.get("voiceTranscript"):
        payload["voiceTranscript"] = entry
    payload.setdefault("visualEmotionTranscript", request_payload.get("visualEmotionTranscript", []) if request_payload else [])

    base_bundle = _fallback_bundle(payload, entry)
    result: Dict[str, Any] = {
        "stress": base_bundle.stress,
        "energy": base_bundle.energy,
        "valence": base_bundle.valence,
        "dominantEmotion": base_bundle.dominant_emotion,
        "overallMood": base_bundle.overall_mood,
        "triggers": base_bundle.triggers,
        "recommendations": base_bundle.recommendations,
        "voiceSummary": base_bundle.voice_summary,
        "emotionSummary": base_bundle.emotion_summary,
        "destinations": base_bundle.destinations,
        "source": "fallback",
    }

    if not allow_gemini:
        return result

    try:
        raw = _call_gemini(payload)
    except Exception:
        return result

    result["source"] = "gemini"
    result["rawResponse"] = raw
    result["overallMood"] = raw.get("overallMood", result["overallMood"])
    result["voiceSummary"] = raw.get("voiceSummary", result["voiceSummary"])
    result["emotionSummary"] = raw.get("emotionSummary", result["emotionSummary"])
    if raw.get("triggers"):
        result["triggers"] = raw["triggers"]
    if raw.get("recommendations"):
        result["recommendations"] = raw["recommendations"]
    if raw.get("destinations"):
        result["destinations"] = raw["destinations"]
    if raw.get("dominantEmotion"):
        result["dominantEmotion"] = raw["dominantEmotion"]
    if "stress" in raw:
        result["stress"] = raw["stress"]
    if "energy" in raw:
        result["energy"] = raw["energy"]
    if "valence" in raw:
        result["valence"] = raw["valence"]

    return result


def _fallback_chat_reply(message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snippet = message.strip().split("\n")[0]
    if len(snippet) > 180:
        snippet = snippet[:177].rstrip() + "..."
    stress_hint = context.get("stress") if isinstance(context, dict) else None
    tone = "grounding" if stress_hint and stress_hint >= 70 else "gentle"
    reply = (
        f"I hear how much is wrapped up in \"{snippet}\". "
        "Let’s take a breath together—notice your shoulders soften a touch. "
        "If you’d like, we can sketch a few experiences that match the feeling you’re naming."
    )
    if tone == "grounding":
        reply = (
            "Thanks for trusting me with that. "
            "Let’s pause for a long exhale and imagine cool air on your skin. "
            "From there we can explore rituals that dial down the pressure while keeping you inspired."
        )
    return {"reply": reply, "source": "fallback"}


def _build_chat_prompt(
    message: str,
    history: List[Dict[str, str]],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [CHAT_STYLE_PROMPT, "", "Conversation so far:"]
    trimmed_history = history[-8:]
    for entry in trimmed_history:
        role = entry.get("role", "user").lower()
        content = entry.get("content", "").strip()
        if not content:
            continue
        prefix = "Traveler" if role == "user" else "Serenity"
        lines.append(f"{prefix}: {content}")
    lines.append(f"Traveler: {message.strip()}")
    if context:
        context_bits = []
        for key in ("stress", "energy", "valence", "dominantEmotion"):
            value = context.get(key)
            if value is None:
                continue
            context_bits.append(f"{key}={value}")
        if context_bits:
            lines.append("")
            lines.append(f"Emotional telemetry: {', '.join(context_bits)}")
    lines.append("")
    lines.append("Respond as Serenity:")
    return "\n".join(lines)


def generate_chat_reply(
    message: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None,
    allow_gemini: bool = True,
) -> Dict[str, Any]:
    if not message or not message.strip():
        raise ValueError("Message is required for chat replies.")

    cleaned_history = history or []
    fallback = _fallback_chat_reply(message, context)
    if not allow_gemini:
        return fallback

    client = _ensure_client()
    if client is None:
        return fallback

    prompt = _build_chat_prompt(message, cleaned_history, context)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
            contents=prompt,
            config={
                "temperature": float(os.getenv("GEMINI_CHAT_TEMPERATURE", os.getenv("GEMINI_TEMPERATURE", "0.85"))),
                "max_output_tokens": int(
                    os.getenv("GEMINI_CHAT_MAX_OUTPUT", os.getenv("GEMINI_MAX_OUTPUT", "512"))
                ),
            },
        )
    except Exception:
        return fallback

    text = getattr(response, "text", None)
    if not text:
        return fallback
    reply = text.strip()
    if not reply:
        return fallback

    return {
        "reply": reply,
        "source": "gemini",
    }