from typing import List, Dict
from nltk.tokenize import word_tokenize as wt


def preproc(s, str_type='None', lower_case=False):
    s = s.replace("\n", ' ')
    if s.strip() == "":
        return ["nothing"]
    if str_type == 'description':
        s = s.split("=-")[1]
    elif str_type == 'inventory':
        s = s.split("carrying")[1]
        if s[0] == ':':
            s = s[1:]
    elif str_type == 'feedback':
        if "Welcome to Textworld" in s:
            s = s.split("Welcome to Textworld")[1]
        if "-=" in s:
            s = s.split("-=")[0]
    s = s.strip()
    if len(s) == 0:
        return ["nothing"]
    tokens = wt(s)
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
