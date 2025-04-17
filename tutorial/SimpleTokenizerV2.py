#!/usr/bin/env python
import re
from termcolor import colored


class SimpleTokenizerV2:
    def __init__(self, data):
        print("...split text into a list:")
        data = re.split(self.regex(), data)
        print(colored(data[0:100], "green"))

        print("...get all unique tokens into a list, building a vocabulary :")
        all_tokens = sorted(list(set(data)))
        print(colored(all_tokens[0:100], "green"))

        print("...append endoftext and unk")
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        print("...clean up spaces with striping")
        preprocessed = [item.strip() for item in all_tokens if item.strip()]
        print(colored(preprocessed[0:100], "green"))

        print("...add numbers to each token/word")
        vocab = {token: integer for integer, token in enumerate(preprocessed)}
        for i in vocab:
            print(" {0} : {1}".format(vocab[i], i))
            if vocab[i] > 30:
                break
        # print(vocab)

        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    @staticmethod
    def regex():
        return r'([,.:;?_!"()\']|--|\s)'

    def encode(self, text):
        text_preprocessed = re.split(self.regex(), text)
        preprocessed = [item.strip() for item in text_preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text


if __name__ == "__main__":
    print("...read the book the-verdict.txt into data")
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        data = file.read()

    print("...init Tokenizer")
    tokenizer = SimpleTokenizerV2(data)

    text = "Hello, world. Is this-- a test?"

    print("...run a test with text: {}".format(text))
    print(colored(tokenizer.encode(text), "green"))
    print(colored(tokenizer.decode(tokenizer.encode(text)), "green"))

    print("...run a test with text book")
    text = data
    print(colored(tokenizer.encode(text)[0:100], "green"))
    print(colored(tokenizer.decode(tokenizer.encode(text)[0:100]), "green"))
