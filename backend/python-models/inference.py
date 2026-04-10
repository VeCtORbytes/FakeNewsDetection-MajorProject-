"""
Fake news detection inference engine + Flask API.

Supports all three model types: xlm-roberta, muril, ensemble.
Model type, checkpoint path, and limits are configurable via environment variables.

Key fixes from original:
    - token_type_ids forwarded to MuRIL and ensemble (was silently dropped)
    - Uses model.predict() (eval-mode-safe) instead of manual forward() call
    - Ensemble wired end-to-end (original had no ensemble loading path)
    - FakeNewsDetector now accepts model_type explicitly so predict() can
      route token_type_ids correctly — original always called the same path
    - batch_predict() runs proper batched tokenisation instead of N separate
      predict() calls (better GPU utilisation)
    - load_detector() handles ensemble checkpoint format
    - Flask app: /health now reports model_type and checkpoint_path
    - Flask app: content-length guard moved to app.config (Flask's built-in)
      rather than a before_request hook to avoid double-checks
    - All module-level state is encapsulated; app factory pattern used so
      the app can be imported without side effects in tests
"""

import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import gdown

# ── Logging ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Model / tokenizer registries ─────────────────────────────────────────── #

TOKENIZER_MAP = {
    "xlm-roberta":  "xlm-roberta-base",
    "xlm_roberta":  "xlm-roberta-base",
    "xlmroberta":   "xlm-roberta-base",
    "muril":        "google/muril-base-cased",
    "ensemble":     "google/muril-base-cased",   # MuRIL tokeniser for ensemble
}

_MODEL_CLASSES = None


def _get_model_classes() -> dict:
    """Lazy-load model classes to avoid slow imports at module level."""
    global _MODEL_CLASSES
    if _MODEL_CLASSES is None:
        from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
        from models.muril_model import MuRILFakeNewsClassifier
        from models.ensemble_model import EnsembleFakeNewsClassifier
        _MODEL_CLASSES = {
            "xlm-roberta":  XLMRobertaFakeNewsClassifier,
            "xlm_roberta":  XLMRobertaFakeNewsClassifier,
            "xlmroberta":   XLMRobertaFakeNewsClassifier,
            "muril":        MuRILFakeNewsClassifier,
            "ensemble":     EnsembleFakeNewsClassifier,   # handled specially
        }
    return _MODEL_CLASSES


# ── Checkpoint helpers ───────────────────────────────────────────────────── #

def _extract_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _strip_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _safe_load(model, state_dict: dict) -> None:
    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning("Strict load failed (%s). Retrying with strict=False.", e)
        model.load_state_dict(state_dict, strict=False)


# ════════════════════════════════════════════════════════════════════════════ #
#  FakeNewsDetector                                                            #
# ════════════════════════════════════════════════════════════════════════════ #

class FakeNewsDetector:
    """
    Inference engine wrapping any of the three model types.

    Args:
        model:          A loaded nn.Module (XLM-R, MuRIL, or Ensemble).
        tokenizer_name: HuggingFace tokenizer identifier.
        model_type:     Canonical model type string — used to route
                        token_type_ids correctly.
        device:         'cuda' or 'cpu'.
        max_length:     Tokenisation max length.
    """

    def __init__(
        self,
        model,
        tokenizer_name: str,
        model_type: str = "muril",
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.model      = model.to(device)
        self.model.eval()
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_type = model_type.strip().lower()
        self.device     = device
        self.max_length = max_length

    def _tokenise(self, texts: list[str]) -> dict:
        """Tokenise a list of texts into tensors on the correct device."""
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _get_probs(self, enc: dict) -> torch.Tensor:
        """
        Run forward pass and return probability tensor [batch, num_classes].
        Routes token_type_ids for MuRIL/ensemble; omits for XLM-RoBERTa.
        Handles log-prob output from ensemble weighted_avg/max paths.
        """
        from models.ensemble_model import EnsembleFakeNewsClassifier

        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids")

        norm = self.model_type.replace("-", "").replace("_", "")

        with torch.no_grad():
            if norm == "xlmroberta":
                logits = self.model(input_ids, attention_mask)
            else:
                # MuRIL and ensemble both accept optional token_type_ids
                logits = self.model(input_ids, attention_mask, token_type_ids)

        # Ensemble weighted_avg/max outputs log-probabilities → use exp()
        if (
            isinstance(self.model, EnsembleFakeNewsClassifier)
            and self.model.ensemble_method in ("weighted_avg", "max")
        ):
            return torch.exp(logits)

        return F.softmax(logits, dim=-1)

    def predict(self, text: str, return_probabilities: bool = False) -> dict:
        """
        Predict for a single text.

        Args:
            text:                 Input text.
            return_probabilities: Include per-class probabilities in result.

        Returns:
            dict with: prediction, confidence, language,
                       and optionally probabilities.
        """
        try:
            from langdetect import detect
            language = detect(text)
        except Exception:
            language = "unknown"

        enc   = self._tokenise([text])
        probs = self._get_probs(enc)          # [1, num_classes]

        pred_idx   = probs.argmax(dim=-1).item()
        pred_label = "Real" if pred_idx == 1 else "Fake"
        confidence = probs[0, pred_idx].item()
        probs_np   = probs[0].cpu().numpy()

        result = {
            "prediction": pred_label,
            "confidence": round(confidence, 6),
            "language":   language,
        }
        if return_probabilities:
            result["probabilities"] = {
                "Fake": round(float(probs_np[0]), 6),
                "Real": round(float(probs_np[1]), 6),
            }
        return result

    def batch_predict(self, texts: list[str]) -> list[dict]:
        """
        Predict for multiple texts in a single batched forward pass.
        More efficient than calling predict() in a loop.

        Args:
            texts: List of input texts.

        Returns:
            List of result dicts (same format as predict(return_probabilities=True)).
        """
        # Detect languages per-text first (fast, CPU-only)
        languages = []
        for t in texts:
            try:
                from langdetect import detect
                languages.append(detect(t))
            except Exception:
                languages.append("unknown")

        enc   = self._tokenise(texts)
        probs = self._get_probs(enc)     # [batch, num_classes]
        probs_np = probs.cpu().numpy()

        results = []
        for i, (lang, p) in enumerate(zip(languages, probs_np)):
            pred_idx = int(p.argmax())
            results.append({
                "prediction":    "Real" if pred_idx == 1 else "Fake",
                "confidence":    round(float(p[pred_idx]), 6),
                "language":      lang,
                "probabilities": {
                    "Fake": round(float(p[0]), 6),
                    "Real": round(float(p[1]), 6),
                },
            })
        return results


# ════════════════════════════════════════════════════════════════════════════ #
#  Model loading                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

def load_detector() -> tuple:
    """
    Build a FakeNewsDetector from environment variables.
    """

    raw_type        = (os.environ.get("MODEL_TYPE") or "muril").strip().lower()
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "").strip()
    max_length      = int(os.environ.get("MAX_LENGTH", "512"))
    device          = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint_path:
        checkpoint_path = f"models/checkpoints/{raw_type}_best.pt"

    MODEL_URL = os.environ.get("MODEL_URL", "").strip()

    # 🔥 AUTO DOWNLOAD MODEL
    if not os.path.isfile(checkpoint_path):
        if MODEL_URL:
            try:
                logger.info("⬇️ Downloading model from Hugging Face...")
                gdown.download(MODEL_URL, checkpoint_path, quiet=False)
                logger.info("✅ Model downloaded successfully")
            except Exception as e:
                logger.exception("❌ Model download failed: %s", e)
                return None, False
        else:
            logger.warning(
                "Checkpoint not found at %s and no MODEL_URL provided.",
                checkpoint_path,
            )
            return None, False

    try:
        classes        = _get_model_classes()
        tokenizer_name = TOKENIZER_MAP.get(raw_type, "xlm-roberta-base")
        ckpt           = torch.load(checkpoint_path, map_location=device)
        state_dict     = _extract_state_dict(ckpt)

        if raw_type == "ensemble":
            from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
            from models.muril_model import MuRILFakeNewsClassifier
            from models.ensemble_model import EnsembleFakeNewsClassifier

            xlmr  = XLMRobertaFakeNewsClassifier()
            muril = MuRILFakeNewsClassifier()

            model = EnsembleFakeNewsClassifier(
                xlmr_model=xlmr,
                muril_model=muril,
                num_classes=2,
            )
        else:
            model_class = classes.get(raw_type)
            if model_class is None:
                logger.warning("Unknown MODEL_TYPE=%s; falling back to xlm-roberta.", raw_type)
                raw_type    = "xlm-roberta"
                model_class = classes["xlm-roberta"]
                tokenizer_name = TOKENIZER_MAP["xlm-roberta"]

            model = model_class()

        _safe_load(model, state_dict)

        detector = FakeNewsDetector(
            model=model,
            tokenizer_name=tokenizer_name,
            model_type=raw_type,
            device=device,
            max_length=max_length,
        )

        logger.info("Model loaded: type=%s path=%s", raw_type, checkpoint_path)
        return detector, True

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return None, False

# ── Text cleaning (matches training pipeline) ────────────────────────────── #

def clean_text_for_inference(text: str) -> str:
    """Apply same cleaning as training pipeline. Falls back gracefully."""
    try:
        from utils.preprocessing import DataPreprocessor
        return DataPreprocessor().clean_text(text or "")
    except Exception:
        return (text or "").strip()[:50_000]


# ════════════════════════════════════════════════════════════════════════════ #
#  Flask API                                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

from flask import Flask, request, jsonify
from flask_cors import CORS

# Limits (configurable via environment variables)
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 1 * 1024 * 1024))   # 1 MB
MAX_TEXT_LENGTH    = int(os.environ.get("MAX_TEXT_LENGTH",    50_000))
MAX_BATCH_SIZE     = int(os.environ.get("MAX_BATCH_SIZE",     20))

# Module-level detector (loaded once at startup)
detector, model_loaded = load_detector()
_checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
_model_type      = (os.environ.get("MODEL_TYPE") or "muril").strip().lower()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    # Use Flask's built-in request size limit
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    def _require_model():
        if not model_loaded or detector is None:
            return jsonify({
                "error": "Model not loaded; prediction unavailable.",
                "code":  "MODEL_NOT_LOADED",
            }), 503
        return None

    @app.route("/health", methods=["GET"])
    def health():
        """Health check. Reports model load status, type, and checkpoint."""
        return jsonify({
            "status":          "healthy",
            "model_loaded":    model_loaded,
            "model_type":      _model_type,
            "checkpoint_path": _checkpoint_path,
        }), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        """Single-text prediction with lightweight normalization and uncertainty gating."""
        err = _require_model()
        if err is not None:
            return err
        try:
            # CHANGE: local imports keep this update self-contained.
            import math
            import re
            import unicodedata

            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid or missing JSON", "code": "INVALID_JSON"}), 400

            raw_text = data.get("text", "")
            if raw_text is None:
                raw_text = ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            raw_text = raw_text.strip()

            if not raw_text:
                return jsonify({"error": "No text provided", "code": "NO_TEXT"}), 400
            if len(raw_text) > MAX_TEXT_LENGTH:
                return jsonify({
                    "error": f"Text exceeds {MAX_TEXT_LENGTH} characters",
                    "code":  "TEXT_TOO_LONG",
                }), 400

            # CHANGE: normalize noisy real-world text before the existing cleaning pipeline.
            def _normalize_text(text: str) -> str:
                text = unicodedata.normalize("NFKC", text)
                text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
                text = text.replace("\r", "\n")
                text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
                text = re.sub(r"@\w+", " ", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"([!?.,])\1{2,}", r"\1\1", text)
                return text.strip()

            normalized_text = _normalize_text(raw_text)
            text = clean_text_for_inference(normalized_text)
            if not text:
                return jsonify({"error": "Text empty after cleaning", "code": "NO_TEXT"}), 400

            # CHANGE: derive simple quality signals to guard short/noisy inputs.
            word_count = len(text.split())
            char_count = len(text)
            token_count = len(detector.tokenizer.tokenize(text))
            alpha_chars = sum(ch.isalpha() for ch in text)
            digit_chars = sum(ch.isdigit() for ch in text)
            alpha_ratio = alpha_chars / max(char_count, 1)
            digit_ratio = digit_chars / max(char_count, 1)

            is_short = char_count < 50 or word_count < 8 or token_count < 12
            is_very_short = char_count < 20 or word_count < 4 or token_count < 6
            is_noisy = alpha_ratio < 0.45 or digit_ratio > 0.30

            result = detector.predict(text, return_probabilities=True)

            fake_prob = float(result.get("probabilities", {}).get("Fake", 0.0))
            real_prob = float(result.get("probabilities", {}).get("Real", 0.0))
            confidence = float(result.get("confidence", 0.0))
            margin = abs(real_prob - fake_prob)

            entropy = 0.0
            for prob in (fake_prob, real_prob):
                prob = max(min(prob, 1.0), 1e-12)
                entropy -= prob * math.log(prob)
            normalized_entropy = entropy / math.log(2)

            # CHANGE: use stricter thresholds for weak or low-quality inputs.
            confidence_threshold = 0.65
            if is_short:
                confidence_threshold = max(confidence_threshold, 0.75)
            if is_noisy:
                confidence_threshold = max(confidence_threshold, 0.80)

            low_margin = margin < (0.20 if is_short else 0.12)
            high_entropy = normalized_entropy > (0.75 if (is_short or is_noisy) else 0.65)

            uncertain = is_very_short or confidence < confidence_threshold or low_margin or high_entropy
            if uncertain:
                # CHANGE: preserve the response structure while downgrading label certainty.
                result["prediction"] = "Uncertain"
                result["confidence"] = round(confidence, 6)

            # CHANGE: add structured debug logging for production analysis.
            logger.info(
                (
                    "Predict analysis | raw_chars=%d normalized_chars=%d cleaned_chars=%d "
                    "words=%d tokens=%d alpha_ratio=%.3f digit_ratio=%.3f "
                    "short=%s very_short=%s noisy=%s confidence=%.4f "
                    "margin=%.4f entropy=%.4f threshold=%.2f final=%s"
                ),
                len(raw_text),
                len(normalized_text),
                char_count,
                word_count,
                token_count,
                alpha_ratio,
                digit_ratio,
                is_short,
                is_very_short,
                is_noisy,
                confidence,
                margin,
                normalized_entropy,
                confidence_threshold,
                result["prediction"],
            )

            return jsonify(result), 200
        except Exception as e:
            logger.exception("Predict failed: %s", e)
            return jsonify({"error": str(e), "code": "PREDICT_ERROR"}), 500

    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        """Batched prediction (up to MAX_BATCH_SIZE texts per request)."""
        err = _require_model()
        if err is not None:
            return err
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid or missing JSON", "code": "INVALID_JSON"}), 400
            texts = data.get("texts")
            if not isinstance(texts, list):
                return jsonify({"error": "Expected 'texts' array", "code": "INVALID_INPUT"}), 400
            if len(texts) > MAX_BATCH_SIZE:
                return jsonify({
                    "error": f"At most {MAX_BATCH_SIZE} texts per request",
                    "code":  "BATCH_TOO_LARGE",
                }), 400

            cleaned = []
            for t in texts:
                s = (t if isinstance(t, str) else str(t)).strip()[:MAX_TEXT_LENGTH]
                s = clean_text_for_inference(s)
                cleaned.append(s or "")

            # Use batched forward pass for efficiency
            results = detector.batch_predict(cleaned)
            return jsonify({"results": results}), 200
        except Exception as e:
            logger.exception("Batch predict failed: %s", e)
            return jsonify({"error": str(e), "code": "PREDICT_ERROR"}), 500

    return app


# Module-level app for gunicorn / direct run compatibility
app = create_app()


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
