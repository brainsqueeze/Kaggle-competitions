import re


def pre_process(text):
    text = text.strip()

    # remove URLs
    text = re.sub(r"^https?://.*[\r\n]*", "", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"http\S+(\s)*(\w+\.\w+)*", "", text, re.MULTILINE | re.IGNORECASE)

    # un-contract
    text = re.sub(r"\'ve", " have ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"cant't", " cannot ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"n't", " not ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"I'm", " I am ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'re", " are ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'d", " would ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'ll", " will ", text, re.MULTILINE | re.IGNORECASE)

    # pad punctuation marks
    text = re.sub(r"([!\"#\$%&\'\(\)\*\+,-\.\/:;\<\=\>\?@\[\\\]\^_`\{\|\}~])", r" \1", text, re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text, re.MULTILINE)

    return text
