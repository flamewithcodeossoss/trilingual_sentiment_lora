from typing import Tuple
import re


class SentimentPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.pipe = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize a transformers sentiment-analysis pipeline if available.

        Falls back to a lightweight rule-based approach if transformers or
        the configured model is unavailable.
        """
        if self.config.get("use_transformers", True):
            try:
                from transformers import pipeline as hf_pipeline
                model_name = self.config.get("model_name")
                self.pipe = hf_pipeline("sentiment-analysis", model=model_name)
            except Exception:
                # Keep self.pipe as None to enable rule-based fallback
                self.pipe = None

    def predict(self, text: str, language: str) -> Tuple[str, float]:
        text = text.strip()
        if not text:
            return "neutral", 0.0

        # Use transformers pipeline if available
        if self.pipe is not None:
            res = self.pipe(text)[0]
            label = str(res.get("label", "")).lower()
            score = float(res.get("score", 0.0))

            # Normalize label to positive/neutral/negative if possible
            if label in ("positive", "negative", "neutral"):
                return label, score

            mapping = self.config.get(
                "label_mapping",
                {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
            )
            return mapping.get(res.get("label", ""), "neutral"), score

        # Fallback to rule-based per-language lexicon
        return self._rule_based(text, language)

    def _rule_based(self, text: str, language: str) -> Tuple[str, float]:
        lex = {
            "en": {
                "pos": {
                    "good",
                    "great",
                    "excellent",
                    "love",
                    "amazing",
                    "happy",
                    "awesome",
                    "wonderful",
                    "nice",
                    "like",
                },
                "neg": {
                    "bad",
                    "terrible",
                    "awful",
                    "hate",
                    "sad",
                    "horrible",
                    "worst",
                    "disappoint",
                    "poor",
                    "angry",
                },
            },
            "ar": {
                "pos": {"جيد", "رائع", "ممتاز", "أحب", "سعيد", "جميل", "مذهل", "لطيف"},
                "neg": {"سيئ", "فظيع", "كريه", "أكره", "حزين", "مزري", "أسوأ", "مخيب", "رديء", "غاضب"},
            },
            "fr": {
                "pos": {"bon", "génial", "excellent", "j'aime", "heureux", "incroyable", "agréable", "super"},
                "neg": {"mauvais", "terrible", "affreux", "je déteste", "triste", "horrible", "pire", "décevant", "pauvre", "fâché"},
            },
        }

        lang = (language or "en").lower()
        d = lex.get(lang, lex["en"])
        words = set(re.findall(r"\w+|\w+'\w+", text.lower()))
        pos = sum(1 for w in words if w in d["pos"])
        neg = sum(1 for w in words if w in d["neg"])

        if pos == neg:
            return "neutral", 0.5
        if pos > neg:
            score = min(1.0, 0.5 + (pos - neg) * 0.1)
            return "positive", score
        score = min(1.0, 0.5 + (neg - pos) * 0.1)
        return "negative", score
        