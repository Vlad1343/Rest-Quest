from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, List, Optional
from queue import Empty

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

try:  # pragma: no cover - optional relative import support
    from .service import generate_analysis, load_mock_request, generate_chat_reply
except ImportError:  # pragma: no cover
    from service import generate_analysis, load_mock_request, generate_chat_reply  # type: ignore


class VisualEmotionEntry(BaseModel):
    question: Optional[int] = None
    prompt: Optional[str] = None
    spectrum: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


class AnalysisRequest(BaseModel):
    entry: Optional[str] = None
    voiceTranscript: Optional[str] = None
    visualEmotionTranscript: List[VisualEmotionEntry] = Field(default_factory=list)
    useMock: bool = False
    allowGemini: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    emotionContext: Optional[Dict[str, Any]] = None
    allowGemini: bool = True


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR / "project"
CAMERA_SCRIPT = PROJECT_DIR / "camera.py"
CAMERA_LOG = PROJECT_DIR / "camera.log"
AUDIO_CACHE_DIR = PROJECT_DIR / "audio_cache"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:  # pragma: no cover - optional dependencies
    from project.camera import (
        EmotionAggregator,
        EmotionVisualizer,
        append_emotion_log,
        cv2 as camera_cv2,
    )
except Exception:  # noqa: BLE001
    EmotionAggregator = None  # type: ignore[assignment]
    EmotionVisualizer = None  # type: ignore[assignment]
    append_emotion_log = None  # type: ignore[assignment]
    camera_cv2 = None  # type: ignore[assignment]

try:  # pragma: no cover
    from project.session_runner import (
        get_session_status,
        start_session,
        subscribe_events,
        unsubscribe_events,
        submit_audio_chunk,
    )
except Exception:  # noqa: BLE001
    get_session_status = None  # type: ignore[assignment]
    start_session = None  # type: ignore[assignment]
    subscribe_events = None  # type: ignore[assignment]
    unsubscribe_events = None  # type: ignore[assignment]

try:  # pragma: no cover
    from project.conversation_runner import (
        get_conversation_status,
        start_conversation,
        request_stop as stop_conversation,
    )
except Exception:  # noqa: BLE001
    get_conversation_status = None  # type: ignore[assignment]
    start_conversation = None  # type: ignore[assignment]
    stop_conversation = None  # type: ignore[assignment]

_camera_process: Optional[subprocess.Popen] = None
_camera_log_handle: Optional[IO[bytes]] = None
_camera_lock = threading.Lock()


app = FastAPI(
    title="Serenity Analysis API",
    version="0.1.0",
    description="Bridges Gemini analysis output with the React frontend.",
)

allowed_origins = os.getenv("ANALYSIS_ALLOWED_ORIGINS", "*").split(",")
origin_list = [origin.strip() for origin in allowed_origins if origin.strip()]
if not origin_list:
    origin_list = ["*"]

allow_credentials = os.getenv("ANALYSIS_ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_request(payload: AnalysisRequest) -> Dict[str, Any]:
    if payload.useMock and not payload.voiceTranscript and not payload.entry:
        return load_mock_request()

    base = payload.model_dump(
        include={"voiceTranscript", "visualEmotionTranscript"},
        exclude_none=True,
    )
    if not base.get("voiceTranscript") and payload.entry:
        base["voiceTranscript"] = payload.entry
    base.setdefault("visualEmotionTranscript", [])
    return base


class CameraCaptureRequest(BaseModel):
    seconds: float = Field(default=6.0, ge=1.0, le=30.0)
    warmup: float = Field(default=1.5, ge=0.0, le=10.0)
    question: Optional[str] = None
    prompt: Optional[str] = None


class ConversationStartRequest(BaseModel):
    turns: int = Field(default=2, ge=1, le=6)


def _camera_stack_available() -> bool:
    return EmotionVisualizer is not None and EmotionAggregator is not None


def _capture_emotions(payload: CameraCaptureRequest) -> Dict[str, Any]:
    if not _camera_stack_available():
        raise RuntimeError("Camera stack unavailable. Install OpenCV/DeepFace dependencies.")

    visualizer = EmotionVisualizer()  # type: ignore[misc]
    aggregator = EmotionAggregator()  # type: ignore[misc]
    frames = 0

    def _collect(duration: float, listening: bool) -> None:
        nonlocal frames
        if duration <= 0:
            return
        end_time = time.time() + duration
        prev = time.time()
        visualizer.is_listening = listening
        while time.time() < end_time:
            _, scores = visualizer.process_frame()
            frames += 1
            now = time.time()
            aggregator.add(scores, now - prev)
            prev = now

    try:
        _collect(payload.warmup, False)
        aggregator.reset()
        _collect(payload.seconds, True)

        spectrum = aggregator.summary()
        dominant = aggregator.dominant()
        timestamp = datetime.utcnow().isoformat()

        if append_emotion_log:
            try:
                append_emotion_log(
                    {
                        "question": payload.question or "React emotional check-in",
                        "transcript": "[camera capture]",
                        "spectrum": spectrum,
                        "dominant": dominant,
                    }
                )
            except Exception:
                pass

        return {
            "dominant": dominant,
            "spectrum": spectrum,
            "frames": frames,
            "seconds": payload.seconds,
            "warmup": payload.warmup,
            "question": payload.question or "React emotional check-in",
            "prompt": payload.prompt or "Camera-guided emotional read",
            "timestamp": timestamp,
        }
    finally:
        try:
            visualizer.is_listening = False
        except Exception:
            pass
        visualizer.release()


def _frame_stream_generator() -> Any:
    if camera_cv2 is None:
        raise RuntimeError("cv2 is unavailable; cannot stream frames.")
    import numpy as _np  # noqa: F401  # ensure numpy present for cv2 encoding

    visualizer = EmotionVisualizer()  # type: ignore[misc]
    try:
        while True:
            frame, _ = visualizer.process_frame()
            ret, buffer = camera_cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        visualizer.release()


def _camera_running() -> bool:
    return _camera_process is not None and _camera_process.poll() is None


def _start_camera_process() -> str:
    global _camera_process, _camera_log_handle

    if not CAMERA_SCRIPT.exists():
        raise RuntimeError(f"Camera script not found at {CAMERA_SCRIPT}")

    if _camera_running():
        return "already_running"

    python_exec = os.getenv("CAMERA_PYTHON", sys.executable)
    CAMERA_LOG.parent.mkdir(parents=True, exist_ok=True)
    _camera_log_handle = CAMERA_LOG.open("ab")

    try:
        _camera_process = subprocess.Popen(
            [python_exec, "-u", str(CAMERA_SCRIPT)],
            cwd=PROJECT_DIR,
            stdout=_camera_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    except Exception:
        _camera_log_handle.close()
        _camera_log_handle = None
        raise

    return "started"


def _stop_camera_process() -> str:
    global _camera_process, _camera_log_handle
    if not _camera_running():
        _camera_process = None
        return "not_running"

    _camera_process.terminate()
    try:
        _camera_process.wait(timeout=5)
        status = "stopped"
    except subprocess.TimeoutExpired:
        _camera_process.kill()
        status = "killed"

    _camera_process = None
    if _camera_log_handle:
        try:
            _camera_log_handle.close()
        except Exception:
            pass
        _camera_log_handle = None
    return status


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/analysis/mock")
async def mock_analysis() -> Dict[str, Any]:
    return generate_analysis(load_mock_request(), allow_gemini=False)


@app.post("/analysis")
async def create_analysis(payload: AnalysisRequest) -> Dict[str, Any]:
    request_dict = _build_request(payload)
    try:
        return generate_analysis(request_dict, entry=payload.entry, allow_gemini=payload.allowGemini)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/respond")
async def chat_respond(payload: ChatRequest) -> Dict[str, Any]:
    try:
        history_payload = [{"role": msg.role, "content": msg.content} for msg in payload.history]
        return generate_chat_reply(
            payload.message,
            history=history_payload,
            context=payload.emotionContext,
            allow_gemini=payload.allowGemini,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/camera/start")
async def start_camera() -> Dict[str, Any]:
    try:
        status = _start_camera_process()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": status}


@app.post("/camera/stop")
async def stop_camera() -> Dict[str, Any]:
    status = _stop_camera_process()
    return {"status": status}


@app.get("/camera/status")
async def camera_status() -> Dict[str, Any]:
    return {"running": _camera_running()}


@app.post("/camera/capture")
async def camera_capture(payload: CameraCaptureRequest) -> Dict[str, Any]:
    if not _camera_stack_available():
        raise HTTPException(status_code=503, detail="Camera stack unavailable on this host.")

    acquired = _camera_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="Camera is busy. Please try again.")

    try:
        return _capture_emotions(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _camera_lock.release()


@app.get("/camera/stream")
async def camera_stream() -> StreamingResponse:
    if not _camera_stack_available():
        raise HTTPException(status_code=503, detail="Camera stack unavailable on this host.")

    acquired = _camera_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="Camera is busy. Please try again.")

    def _generator():
        try:
            yield from _frame_stream_generator()
        finally:
            _camera_lock.release()

    return StreamingResponse(_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/conversation/start")
async def conversation_start(payload: ConversationStartRequest) -> Dict[str, Any]:
    if not start_conversation:
        raise HTTPException(status_code=503, detail="Conversation runner unavailable on this host.")
    try:
        start_conversation(turns=payload.turns)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.get("/conversation/status")
async def conversation_status() -> Dict[str, Any]:
    if not get_conversation_status:
        raise HTTPException(status_code=503, detail="Conversation runner unavailable on this host.")
    return get_conversation_status()


@app.post("/conversation/stop")
async def conversation_stop() -> Dict[str, Any]:
    if not stop_conversation:
        raise HTTPException(status_code=503, detail="Conversation runner unavailable on this host.")
    stop_conversation()
    return {"status": "stopping"}


@app.post("/session/start")
async def session_start() -> Dict[str, Any]:
    if not start_session:
        raise HTTPException(status_code=503, detail="Session runner unavailable on this host.")
    try:
        start_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.get("/session/status")
async def session_status() -> Dict[str, Any]:
    if not get_session_status:
        raise HTTPException(status_code=503, detail="Session runner unavailable on this host.")
    return get_session_status()


@app.get("/session/events")
async def session_events() -> StreamingResponse:
    if not subscribe_events or not unsubscribe_events:
        raise HTTPException(status_code=503, detail="Session runner unavailable on this host.")

    subscriber = subscribe_events()

    def _event_generator():
        try:
            while True:
                try:
                    event = subscriber.get(timeout=1.0)
                    payload = f"data: {json.dumps(event)}\n\n"
                    yield payload.encode("utf-8")
                except Empty:
                    yield b": keep-alive\n\n"
        finally:
            unsubscribe_events(subscriber)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post("/session/audio")
async def session_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not submit_audio_chunk:
        raise HTTPException(status_code=503, detail="Session audio handler unavailable on this host.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload.")
    submit_audio_chunk(data)
    return {"status": "queued"}


@app.get("/audio/prompts/{filename}")
async def get_prompt_audio(filename: str):
    path = AUDIO_CACHE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop_camera_process()