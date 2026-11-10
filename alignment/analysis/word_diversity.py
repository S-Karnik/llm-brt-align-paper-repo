from nltk import word_tokenize

def compute_rep_n(text: str, n: int) -> float:
    tokens = word_tokenize(text)
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n


def compute_diversity(text: str) -> float:
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n_val / 100
    return diversity

if __name__ == "__main__":
    texts = [
        "This is a test. This test is only a test.",
        "Diversity in text is important. Unique phrases make text interesting.",
        "Hello world! Hello universe! Hello everyone!",
    ]
    for text in texts:
        div = compute_diversity(text)
        print(f"Text: {text}\nDiversity: {div:.4f}\n")
