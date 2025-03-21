from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import urllib.request

def download_vocab(vocab_path="vocab.txt", vocab_size=10000):
    if os.path.exists(vocab_path):
        return

    print(f"🔤 Downloading top {vocab_size} common English words for vocab...")

    # Download top 10k common English words list
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
    tmp_path = "common_words_tmp.txt"
    urllib.request.urlretrieve(url, tmp_path)

    # Filter first vocab_size words, remove short or weird symbols
    with open(tmp_path, "r") as f:
        words = [line.strip().lower() for line in f if line.strip().isalpha() and len(line.strip()) > 2]

    clean_words = words[:vocab_size]

    with open(vocab_path, "w") as f:
        f.write("\n".join(clean_words))

    os.remove(tmp_path)
    print(f"✅ Saved cleaned vocab to {vocab_path} ({len(clean_words)} words).")

# Load or create vocab
VOCAB_PATH = "vocab.txt"
download_vocab(VOCAB_PATH)

# Load vocab
with open(VOCAB_PATH, "r") as f:
    vocabulary = [line.strip() for line in f if line.strip()]

# Load sentence embedding model
print("🧠 Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded.")

def get_mean_vector(words):
    if not words:
        return None
    return model.encode(words, convert_to_numpy=True).mean(axis=0)

def get_codenames_clue(
    positive_words,
    unrelated_words=None,
    enemy_words=None,
    assassin_word=None,
    w_unrelated=0.5,
    w_enemy=1.0,
    w_assassin=3.0,
    topn=5,
):
    unrelated_words = unrelated_words or []
    enemy_words = enemy_words or []

    vec_pos = get_mean_vector(positive_words)
    # vec_unrel = get_mean_vector(unrelated_words)
    # vec_enemy = get_mean_vector(enemy_words)
    # vec_assassin = model.encode([assassin_word])[0] if assassin_word else None

    if vec_pos is None:
        return "❌ None of your team words are valid."

    clue_vec = vec_pos
    # if vec_unrel is not None:
    #     clue_vec -= w_unrelated * vec_unrel
    # if vec_enemy is not None:
    #     clue_vec -= w_enemy * vec_enemy
    # if vec_assassin is not None:
    #     clue_vec -= w_assassin * vec_assassin

    vocab_vectors = model.encode(vocabulary, convert_to_numpy=True)
    sims = cosine_similarity([clue_vec], vocab_vectors)[0]

    sorted_indices = np.argsort(sims)[::-1]
    forbidden = set(positive_words + unrelated_words + enemy_words + ([assassin_word] if assassin_word else []))

    results = []
    for idx in sorted_indices:
        word = vocabulary[idx]
        if word.lower() not in forbidden and all([bad_word not in word.lower() for bad_word in forbidden]):
            results.append((word, sims[idx]))
        if len(results) >= topn:
            break

    return results

if __name__ == "__main__":
    print("\n🎯 Enter your team words (e.g., king queen knight):")
    positive = input("Your team words: ").lower().split()

    # print("\n🙃 Enter unrelated/neutral words to avoid (optional):")
    # unrelated = input("Unrelated words: ").lower().split()

    # print("\n⚠ Enter enemy team words (to avoid pointing to):")
    # enemy = input("Enemy words: ").lower().split()

    # print("\n💀 Enter THE assassin word (the deadly one to absolutely avoid):")
    # assassin = input("Assassin word: ").strip().lower()

    clues = get_codenames_clue(positive)
    print("\n💡 Suggested Clues:")
    for word, score in clues:
        print(f"→ {word} (score: {score:.3f})")
