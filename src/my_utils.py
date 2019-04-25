import random
from string import ascii_uppercase, digits, punctuation
import numpy


def robust_padding(texts, labels):
    maxlen = max(len(t) for t in texts)

    for i, text in enumerate(texts):
        if len(text) == maxlen:
            continue

        pad_before = random.randint(0, maxlen - len(text))
        pad_after = maxlen - pad_before - len(text)

        texts[i] = random_string(pad_before) + text + random_string(pad_after)
        labels[i] = numpy.pad(labels[i], (pad_before, pad_after), "constant", constant_values=0)


def random_string(n):
    if n == 0:
        return ""

    x = random.random()
    if x > 0.5:
        pad = " " * n
    elif x > 0.3:
        pad = "".join(random.choices(digits + " \t\n", k=n))
    elif x > 0.2:
        pad = "".join(random.choices(ascii_uppercase + " \t\n", k=n))
    elif x > 0.1:
        pad = "".join(random.choices(ascii_uppercase + digits + " \t\n", k=n))
    else:
        pad = "".join(random.choices(ascii_uppercase + digits + punctuation + " \t\n", k=n))

    return pad


if __name__ == "__main__":
    x = ["be", "diudui", "debdbuyubuqqo"]
    y = [numpy.ones(len(xi)) for xi in x]
    robust_padding(x, y)

    print(x)
    print(y)
