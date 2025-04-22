#!/usr/bin/env python
import re, time
import torch
from termcolor import colored


class SimpleTokenizerV2:
    def __init__(self, corpus_path="corpus_text/the-verdict.txt"):
        self.sleep()
        print("...read the corpus text path {}".format(corpus_path))
        with open(corpus_path, "r", encoding="utf-8") as file:
            self.corpus_text = file.read()

        self.sleep()
        print("...split text into a list:")

        self.data = re.split(self.regex(), self.corpus_text)
        print(colored(self.data[0:100], "green"))

        self.sleep()
        print("...get all unique tokens into a list, building a vocabulary :")
        all_tokens = sorted(list(set(self.data)))
        print(colored(all_tokens[0:100], "green"))

        self.sleep()
        print("...append endoftext and unk")
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        self.sleep()
        print("...clean up spaces with striping")
        preprocessed = [item.strip() for item in all_tokens if item.strip()]
        print(colored(preprocessed[0:100], "green"))

        self.sleep()
        print("...add numbers to each token/word")
        vocab = {token: integer for integer, token in enumerate(preprocessed)}
        for i in vocab:
            print(" {0} : {1}".format(vocab[i], i))
            if vocab[i] > 15:
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

    def sleep(self, seconds=1):
        time.sleep(seconds)


if __name__ == "__main__":
    print("...Machine has GPU: ", torch.cuda.is_available())

    print("...init Tokenizer")
    t = SimpleTokenizerV2("corpus_text/the-verdict.txt")

    t.sleep()
    text = "Hello, world. Is this-- a test?"
    print("...run a test with text: {}".format(text))
    print(colored(t.encode(text), "green"))
    print(colored(t.decode(t.encode(text)), "green"))

    t.sleep()
    print("...run a test with text book")
    text = t.corpus_text
    print(colored(t.encode(text)[0:100], "green"))
    print(colored(t.decode(t.encode(text)[0:100]), "green"))
