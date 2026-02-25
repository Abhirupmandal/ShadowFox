from spellchecker import SpellChecker

class AutoCorrectEngine:
    def __init__(self):
        self.spell = SpellChecker()

    def correct(self, text):
        words = text.split()
        corrected = [self.spell.correction(w) for w in words]
        return " ".join(corrected)