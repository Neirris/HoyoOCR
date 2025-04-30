import itertools
import random
import argparse
import os
from collections import defaultdict


def generate_corpus(
    output_file, letters="abcdefghijklmnopqrstuvwxyz", min_len=1, max_len=6, samples=100
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    letter_counts = defaultdict(int)
    generated_words = set()

    def add_word(word):
        generated_words.add(word)
        for letter in word:
            letter_counts[letter] += 1

    with open(output_file, "w") as f:
        for length in range(min_len, max_len + 1):
            try:
                if length <= 2:
                    for combo in itertools.product(letters, repeat=length):
                        word = "".join(combo)
                        if word not in generated_words:
                            f.write(word + "\n")
                            add_word(word)
                else:
                    effective_length = min(length, len(letters))
                    for _ in range(samples):
                        combo = random.sample(letters, effective_length)
                        word = "".join(combo)
                        if word not in generated_words:
                            f.write(word + "\n")
                            add_word(word)
            except ValueError:
                continue

        missing_letters = {letter for letter in letters if letter_counts[letter] < 10}

        while missing_letters:
            letter = missing_letters.pop()

            needed = 10 - letter_counts[letter]
            generated = 0

            while generated < needed:
                length = random.randint(1, max_len)

                if length == 1:
                    word = letter
                else:
                    other_letters = [L for L in letters if L != letter]
                    other_length = min(length - 1, len(other_letters))
                    other_chars = random.sample(other_letters, other_length)

                    chars = [letter] + other_chars
                    random.shuffle(chars)
                    word = "".join(chars)

                if word not in generated_words:
                    f.write(word + "\n")
                    add_word(word)
                    generated += 1

            missing_letters = {
                letter for letter in letters if letter_counts[letter] < 10
            }


def main():
    parser = argparse.ArgumentParser()
    default_output = os.path.join("assets", "misc", "corpus.txt")

    parser.add_argument(
        "--output",
        help="default: assets/misc/corpus.txt",
        default=default_output,
    )
    parser.add_argument("--letters", default="abcdefghijklmnopqrstuvwxyz")
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--samples", type=int, default=100)

    args = parser.parse_args()

    if args.min_len < 1:
        args.min_len = 1

    if args.max_len < args.min_len:
        args.max_len = args.min_len

    generate_corpus(
        output_file=args.output,
        letters=args.letters,
        min_len=args.min_len,
        max_len=args.max_len,
        samples=args.samples,
    )


if __name__ == "__main__":
    main()
