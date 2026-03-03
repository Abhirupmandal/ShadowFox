from spellchecker import SpellChecker
import difflib

class AutoCorrectEngine:

    def __init__(self):
        self.spell = SpellChecker()

    def correct(self, text):
        words = text.split()
        corrected = []

        for word in words:

            suggestion = self.spell.correction(word)

            if suggestion is None:
                matches = difflib.get_close_matches(
                    word,
                    self.spell.word_frequency.words(),
                    n=1,
                    cutoff=0.8
                )
                suggestion = matches[0] if matches else word

            corrected.append(suggestion)

        return " ".join(corrected)