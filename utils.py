import jieba
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def calculate_bleu(hypothesis, targets, cut=True, verbose=False):
    bleu_scores = []
    for sent, trg in zip(hypothesis, targets):
        trg = trg.strip().lower().split()
        sent = sent.strip()
        if cut:
            trg = list(jieba.cut(''.join(trg).replace("-", "")))
            sent = list(jieba.cut(''.join(sent).replace("-", "")))

        bleu = sentence_bleu([trg], sent, weights=(0.5, 0.5, 0., 0.), smoothing_function=SmoothingFunction().method1)
        if verbose:
            print(f"src:{sent.strip()}\ntrg:{trg}\npredict:{sent}\n{bleu}\n")
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)