from src.config import LOG_PATH

def log(text):
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")