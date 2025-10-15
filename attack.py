# pip install fastapi uvicorn transformers torch pydantic accelerate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Victim Model Server", version="1.0.0")

# ----- Models -----
class ChatRequest(BaseModel):
    model: str
    message: str
    max_length: int = 100
    temperature: float = 0.7

class MessageContent(BaseModel):
    role: str = "assistant"
    content: str

class ChatResponseData(BaseModel):
    success: bool = True
    message: MessageContent
    tokens_in: int = 0
    tokens_out: int = 0

class ChatResponse(BaseModel):
    response: ChatResponseData

# ----- Victim Model Setup -----
class VictimModel:
    def __init__(self, model_id: str = "microsoft/DialoGPT-small"):
        """
        Initialize a small, lightweight conversational model.
        DialoGPT-small is ~117MB, perfect for testing.
        
        Other lightweight options:
        - "microsoft/DialoGPT-medium" (~355MB)
        - "distilgpt2" (~82MB, less conversational)
        - "facebook/opt-125m" (~250MB)
        """
        logger.info(f"Loading victim model: {model_id}...")
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_id = model_id
            
            logger.info(f"Victim model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load victim model: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Dictionary with generated text and token counts
        """
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt + self.tokenizer.eos_token,
                return_tensors="pt",
                # truncate=True,
                max_length=512
            ).to(self.device)
            
            tokens_in = inputs.shape[1]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(max_length + tokens_in, 512),
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the new generated part (remove input prompt)
            if generated_text.startswith(prompt):
                response_text = generated_text[len(prompt):].strip()
            else:
                response_text = generated_text.strip()
            
            # Fallback if generation failed
            if not response_text or len(response_text) < 3:
                response_text = "I understand. How can I help you with that?"
            
            tokens_out = outputs.shape[1] - tokens_in
            latency_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Generated response ({tokens_out} tokens, {latency_ms}ms)")
            
            return {
                "text": response_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "text": "I apologize, but I encountered an error processing your request.",
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "error": str(e)
            }

# Initialize victim models (can support multiple)
# Model registry: model_id -> VictimModel instance
MODELS: Dict[str, VictimModel] = {}

def get_or_load_model(model_id: str) -> VictimModel:
    """Get cached model or load new one."""
    if model_id not in MODELS:
        # Map model IDs to actual model names
        model_map = {
            "0": "microsoft/DialoGPT-small",
            "1": "microsoft/DialoGPT-medium",
            "2": "distilgpt2"
        }
        
        actual_model = model_map.get(model_id, "microsoft/DialoGPT-small")
        logger.info(f"Loading model {model_id} -> {actual_model}")
        MODELS[model_id] = VictimModel(actual_model)
    
    return MODELS[model_id]

# ----- API Endpoints -----
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "victim-model-server",
        "loaded_models": list(MODELS.keys()),
        "device": next(iter(MODELS.values())).device if MODELS else "none"
    }

@app.post("/chat/respondTo", response_model=ChatResponse)
async def generate_text(request: ChatRequest):
    """
    Main endpoint for text generation.
    
    Args:
        request: ChatRequest with model ID, message, and generation params
    
    Returns:
        ChatResponse with generated text and metadata
    """
    try:
        logger.info(f"Request for model '{request.model}': {request.message[:50]}...")
        
        # Get or load the requested model
        model = get_or_load_model(request.model)
        
        # Generate response
        result = model.generate_response(
            prompt=request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Check for errors
        if "error" in result:
            logger.error(f"Generation error: {result['error']}")
            return ChatResponse(
                response=ChatResponseData(
                    success=False,
                    message=MessageContent(
                        role="assistant",
                        content=result["text"]
                    ),
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0)
                )
            )
        
        # Return successful response
        return ChatResponse(
            response=ChatResponseData(
                success=True,
                message=MessageContent(
                    role="assistant",
                    content=result["text"]
                ),
                tokens_in=result["tokens_in"],
                tokens_out=result["tokens_out"]
            )
        )
    
    except Exception as e:
        logger.error(f"Error in generate_text endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_simple(request: ChatRequest):
    """
    Alternative endpoint with simpler response format.
    """
    try:
        model = get_or_load_model(request.model)
        result = model.generate_response(
            prompt=request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return {
            "model": request.model,
            "prompt": request.message,
            "response": result["text"],
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"],
            "latency_ms": result.get("latency_ms", 0)
        }
    
    except Exception as e:
        logger.error(f"Error in generate_simple endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "loaded_models": list(MODELS.keys()),
        "available_models": [
            "0", "1", "2", "victim1", "victim2",
            "small", "medium", "distilgpt2"
        ]
    }

# ----- Startup Event -----
@app.on_event("startup")
async def startup_event():
    logger.info("Victim Model Server started")
    logger.info("Pre-loading default model...")
    
    # Pre-load one model to speed up first request
    try:
        get_or_load_model("0")
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load default model: {e}")

# ----- Main -----
if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from defense server
        log_level="info"
    )