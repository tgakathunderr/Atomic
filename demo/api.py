import os
import sys
import time
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from atomic.generate import generate_stream, GenerationConfig

from contextlib import asynccontextmanager

# Global model state
MODEL_DIR = os.environ.get("ATOMIC_MODEL_DIR", "models/atomic-reasoning-prod")
config: Optional[AtomicConfig] = None
tokenizer: Optional[AtomicTokenizer] = None
model: Optional[AtomicForCausalLM] = None


def load_model():
    global config, tokenizer, model
    config_path = os.path.join(MODEL_DIR, "config.json")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    weights_path = os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found: {config_path}")

    config = AtomicConfig.from_json(config_path)
    tokenizer = AtomicTokenizer.from_pretrained(tokenizer_path)
    model = AtomicForCausalLM(config)

    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(weights)

    model.eval()
    print(f"Loaded ATOMIC model from {MODEL_DIR} ({model.num_parameters():,} parameters)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="ATOMIC Reasoning Inference API",
    description="High-throughput CPU Recurrent LM serving with constant O(1) memory footprint.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "What is 45 + 55?"})
    max_new_tokens: int = Field(default=120, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.8, le=2.0)
    stream: bool = Field(default=False)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 120
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "atomic-reasoning-prod",
        "architecture": "ATOMIC (Gated Linear Recurrent Attention)",
        "parameters": model.num_parameters() if model else 0,
        "cache_memory_per_stream": "64.0 KB fixed",
        "hardware": "CPU (O(1) Recurrent Execution)",
        "device": "cpu"
    }


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Format reasoning prompt
    prompt = req.prompt
    if "<think>" not in prompt and not prompt.endswith("\n"):
        prompt = f"Question: {prompt}\n<think>\n"

    gen_cfg = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop_strings=["</answer>", "<eos>"]
    )

    if req.stream:
        def event_stream():
            stream = generate_stream(model, tokenizer, prompt, gen_cfg)
            for chunk in stream:
                data = json.dumps({"token": chunk})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming response
    t0 = time.perf_counter()
    stream = generate_stream(model, tokenizer, prompt, gen_cfg)
    completion = "".join(list(stream))
    total_time = time.perf_counter() - t0

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": prompt + completion,
        "generation_time_seconds": round(total_time, 4),
        "recurrent_state_footprint": "64 KB"
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Extract last user message
    user_msg = ""
    for msg in reversed(req.messages):
        if msg.role.lower() == "user":
            user_msg = msg.content
            break

    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found in conversation")

    prompt = f"Question: {user_msg}\n<think>\n"

    gen_cfg = GenerationConfig(
        max_new_tokens=req.max_tokens or 120,
        temperature=req.temperature or 0.0,
        repetition_penalty=1.1,
        stop_strings=["</answer>", "<eos>"]
    )

    if req.stream:
        def sse_stream():
            stream = generate_stream(model, tokenizer, prompt, gen_cfg)
            for chunk in stream:
                payload = {
                    "id": "atomic-chat",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "atomic-reasoning-prod",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    stream = generate_stream(model, tokenizer, prompt, gen_cfg)
    content = "".join(list(stream))

    return {
        "id": "atomic-chat",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "atomic-reasoning-prod",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
