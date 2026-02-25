from core.autocorrect_engine import AutoCorrectEngine
from core.language_model_engine import LanguageModelEngine
from core.text_processing import normalize_text
from core.cache import PredictionCache
from core.config import CACHE_SIZE

spell_engine = AutoCorrectEngine()
lm_engine = LanguageModelEngine()
cache = PredictionCache(CACHE_SIZE)

class KeyboardService:

    @staticmethod
    def process(text):

        text = normalize_text(text)

        cached = cache.get(text)
        if cached:
            return cached

        corrected = spell_engine.correct(text)
        predictions = lm_engine.predict(corrected)

        cache.put(text, (corrected, predictions))

        return corrected, predictions