# pip install fastapi uvicorn transformers torch pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re
import os 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Defense Model Server", version="1.0.0")

# ----- Models -----
class ChatRequest(BaseModel):
    model: str
    message: str
    analyze_only: bool = False

class DecisionResponse(BaseModel):
    action: Literal["allow", "rewrite", "block"]
    reason: str
    rewritten_prompt: Optional[str] = None
    risk_score: float = Field(ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: DecisionResponse

# ----- Defense Model Setup -----
class DefenseModel:
    def __init__(self):
        """
        Initialize a small, efficient model for prompt analysis.
        Using google-t5 for text classification as example.
        """
        logger.info("Loading defense model...")

        model_name = "google-t5/t5-small"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # For demo, we'll use a simple classification approach
            # In production, use a model fine-tuned on jailbreak/injection detection
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # For this demo, we'll use rule-based + simple heuristics
            # You can replace this with actual model inference
            self.use_ml_model = False  # Set to True when you have a fine-tuned model
            
            if self.use_ml_model:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            
            logger.info("Defense model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Define attack patterns
        self.attack_patterns = [
            # Prompt injection patterns
            (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", "prompt_injection"),
            (r"disregard\s+(all\s+)?(previous|prior|above)", "prompt_injection"),
            (r"forget\s+(all\s+)?(previous|prior|above)", "prompt_injection"),
            (r"new\s+instructions?:", "prompt_injection"),
            (r"system\s*:\s*you\s+are", "role_manipulation"),
            (r"your\s+new\s+role\s+is", "role_manipulation"),
            (r"act\s+as\s+(if\s+)?you\s+(are|were)", "role_manipulation"),
            (r"pretend\s+(you\s+are|to\s+be)", "role_manipulation"),
            
            # Jailbreak attempts
            (r"(DAN|AIM|DUDE)\s+mode", "jailbreak"),
            (r"opposite\s+mode", "jailbreak"),
            (r"evil\s+(mode|assistant|AI)", "jailbreak"),
            (r"without\s+any\s+(moral|ethical)", "jailbreak"),
            (r"no\s+(ethical|moral)\s+guidelines", "jailbreak"),
            
            # Information extraction
            (r"(reveal|show|tell|give)\s+(me\s+)?(your|the)\s+(system|secret|password|key|prompt)", "info_extraction"),
            (r"what\s+(is|are)\s+your\s+(instructions|system\s+prompt|rules)", "info_extraction"),
            (r"repeat\s+(your|the)\s+(instructions|prompt)", "info_extraction"),
            
            # Harmful content requests
            (r"how\s+to\s+(make|create|build)\s+(a\s+)?(bomb|weapon|explosive)", "harmful_content"),
            (r"(kill|harm|hurt|murder)\s+(someone|people|a\s+person)", "harmful_content"),
            (r"steal\s+(credit\s+card|password|identity)", "harmful_content"),
            
            # Encoding tricks
            (r"base64|rot13|hex\s+encoded|unicode\s+escape", "encoding_obfuscation"),
            (r"\\u[0-9a-fA-F]{4}", "encoding_obfuscation"),
            
            # Multi-turn attacks
            (r"step\s+\d+\s*:\s*ignore", "multi_turn_attack"),
            (r"phase\s+\d+\s*:\s*forget", "multi_turn_attack"),
        ]
        
        # Suspicious keywords
        self.suspicious_keywords = [
            "jailbreak", "bypass", "override", "exploit", "vulnerability",
            "hack", "manipulate", "trick", "deceive", "fool", "circumvent"
        ]
    
    def analyze_prompt(self, prompt: str) -> DecisionResponse:
        """
        Analyze a prompt for malicious intent.
        Returns a decision: allow, rewrite, or block.
        """
        prompt_lower = prompt.lower()
        
        # Track detected issues
        detected_attacks = []
        risk_factors = []
        risk_score = 0.0
        
        # Check for attack patterns
        for pattern, attack_type in self.attack_patterns:
            if re.search(pattern, prompt_lower):
                detected_attacks.append(attack_type)
                risk_score += 0.3
                logger.warning(f"Detected {attack_type}: {pattern}")
        
        # Check for suspicious keywords
        for keyword in self.suspicious_keywords:
            if keyword in prompt_lower:
                risk_factors.append(keyword)
                risk_score += 0.1
        
        # Check for excessive special characters (potential obfuscation)
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', prompt)) / max(len(prompt), 1)
        if special_char_ratio > 0.3:
            risk_factors.append("high_special_char_ratio")
            risk_score += 0.2
        
        # Check for very long prompts (potential overflow attacks)
        if len(prompt) > 2000:
            risk_factors.append("excessive_length")
            risk_score += 0.1
        
        # Check for repeated instructions
        instruction_count = len(re.findall(r'\b(instruction|command|directive|order)\b', prompt_lower))
        if instruction_count > 3:
            risk_factors.append("repeated_instructions")
            risk_score += 0.15
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Decision logic
        if risk_score >= 0.7 or len(detected_attacks) >= 2:
            # HIGH RISK - Block completely
            return DecisionResponse(
                action="block",
                reason=f"High risk detected: {', '.join(set(detected_attacks))}",
                risk_score=risk_score
            )
        
        elif risk_score >= 0.3 or len(detected_attacks) == 1:
            # MEDIUM RISK - Rewrite/sanitize
            sanitized = self._sanitize_prompt(prompt, detected_attacks)
            return DecisionResponse(
                action="rewrite",
                reason=f"Potential attack sanitized: {', '.join(set(detected_attacks + risk_factors))}",
                rewritten_prompt=sanitized,
                risk_score=risk_score
            )
        
        else:
            # LOW RISK - Allow
            return DecisionResponse(
                action="allow",
                reason="No significant threats detected",
                risk_score=risk_score
            )
    
    def _sanitize_prompt(self, prompt: str, detected_attacks: list) -> str:
        """
        Sanitize a prompt by removing or neutralizing malicious elements.
        """
        sanitized = prompt
        
        # Remove common attack patterns
        patterns_to_remove = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?[.!]*",
            r"disregard\s+(all\s+)?(previous|prior|above)[.!]*",
            r"forget\s+(all\s+)?(previous|prior|above)[.!]*",
            r"new\s+instructions?:[^\n]*",
            r"system\s*:\s*you\s+are[^\n]*",
            r"your\s+new\s+role\s+is[^\n]*",
        ]
        
        for pattern in patterns_to_remove:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove excessive special characters
        sanitized = re.sub(r'[\\]{2,}', '', sanitized)
        
        # Trim and clean up
        sanitized = ' '.join(sanitized.split())
        sanitized = sanitized.strip()
        
        # If sanitization removed too much, return a safe default
        if len(sanitized) < 10:
            sanitized = "[Sanitized query - original content removed due to security concerns]"
        
        # Add safety prefix
        sanitized = f"[SANITIZED] {sanitized}"
        
        return sanitized

# Initialize defense model
defense = DefenseModel()

# ----- API Endpoints -----
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "defense-model-server",
        "device": defense.device if hasattr(defense, 'device') else "unknown"
    }

@app.post("/chat/respondTo", response_model=ChatResponse)
async def analyze_prompt(request: ChatRequest):
    """
    Main endpoint for analyzing prompts.
    
    Args:
        request: ChatRequest with model, message, and analyze_only flag
    
    Returns:
        ChatResponse with decision (allow/rewrite/block), reason, and optional rewritten prompt
    """
    try:
        logger.info(f"Analyzing prompt (length: {len(request.message)})")
        
        # Analyze the prompt
        decision = defense.analyze_prompt(request.message)
        
        logger.info(f"Decision: {decision.action} (risk: {decision.risk_score:.2f})")
        
        return ChatResponse(response=decision)
    
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        # On error, default to blocking for safety
        return ChatResponse(
            response=DecisionResponse(
                action="block",
                reason=f"Analysis error: {str(e)}",
                risk_score=1.0
            )
        )

@app.post("/analyze")
async def analyze_only(request: ChatRequest):
    """
    Dedicated analysis endpoint (same as respondTo but more explicit).
    """
    return await analyze_prompt(request)

# ----- Startup Event -----
@app.on_event("startup")
async def startup_event():
    logger.info("Defense Model Server started")
    logger.info("Ready to analyze prompts")

# ----- Main -----
if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )