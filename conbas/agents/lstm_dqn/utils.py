from typing import List, Dict


def preprocess(input_str: str, tokenizer, is_observation=False, lower_case=True) -> List[str]:
    """Preprocess input string and tokenize it."""
    if input_str is None:
        return ["nothing"]

    input_str = input_str.replace("\n", ' ')
    if input_str.strip() == "":
        return ["nothing"]

    if is_observation:
        if "$$$$$$$" in input_str:
            input_str = ""
        if "-=" in input_str:
            input_str = input_str.split("-=")[0]

    input_str = input_str.strip()
    if len(input_str) == 0:
        return ["nothing"]

    tokens = [t.text for t in tokenizer(input_str)]

    if lower_case:
        tokens = [t.lower() for t in tokens]

    return tokens


def words_to_ids(words: List[str], word2id: Dict[str, int]) -> List[int]:
    ids = []
    for word in words:
        try:
            ids.append(word2id[word])
        except KeyError:
            ids.append(word2id["<UNK>"])

    return ids


def linear_decay_fn(upper_bound, lower_bound, duration):
    return lambda step:  max(lower_bound, upper_bound - (upper_bound - lower_bound) * step / duration)


def linear_inc_fn(upper_bound, lower_bound, duration):
    return lambda step: min(upper_bound, lower_bound + (upper_bound - lower_bound) * step / duration)
