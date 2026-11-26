"""Pure-Python CTC prefix beam search with a word n-gram language model (ARPA). No compiled dependencies.

Character CTC (blank = index 0, space = ' ') + word LM, as in Hannun et al. 2014 ("first-pass LVCSR with CTC + LM"):
  score(prefix) = log P_ctc(prefix | audio) + alpha * ln P_lm(words) + beta * (number of words)
A word gets its LM score when it is completed (space or end of audio). A partial word that is not the beginning of any
LM word gets a penalty (alpha * UNK_LOG10 * ln 10), so the search prefers real words.
"""
from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path

import numpy as np

LN10 = math.log(10.0)
NEG = -1e30
UNK_LOG10 = -10.0                      # log10 score for a word the LM does not know (as pyctcdecode's default offset)


def _lse(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


class NgramLM:
    """Back-off n-gram LM from an ARPA file. Words are upper case (LibriSpeech LMs are)."""

    def __init__(self, prob: dict, bow: dict, order: int, prefixes: set):
        self.prob, self.bow, self.order, self.prefixes = prob, bow, order, prefixes
        self.extra = {}                                  # hotwords: word -> log10 unigram score

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_arpa(cls, path: str | Path, log=print) -> "NgramLM":
        path = Path(path)
        op = gzip.open if path.suffix == ".gz" else open
        prob, bow, order, sec = {}, {}, 0, 0
        with op(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("\\"):
                    if line.endswith("-grams:"):
                        sec = int(line[1:line.index("-")]); order = max(order, sec)
                    else:
                        sec = 0
                    continue
                if sec == 0:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split()
                    lp, words, bo = float(parts[0]), " ".join(parts[1:1 + sec]), (float(parts[1 + sec]) if len(parts) > 1 + sec else None)
                else:
                    lp, words, bo = float(parts[0]), parts[1], (float(parts[2]) if len(parts) > 2 else None)
                prob[words] = lp
                if bo is not None and bo != 0.0:
                    bow[words] = bo
        lm = cls(prob, bow, order, cls._prefixes(prob))
        log(f"  ARPA loaded: order {order}, {len(prob):,} n-grams, {sum(1 for w in prob if ' ' not in w):,} words")
        return lm

    @staticmethod
    def _prefixes(prob):
        out = set()
        for w in prob:
            if " " not in w and not w.startswith("<"):
                for k in range(1, len(w) + 1):
                    out.add(w[:k])
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(prob=self.prob, bow=self.bow, order=self.order), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["prob"], d["bow"], d["order"], cls._prefixes(d["prob"]))

    @classmethod
    def load_cached(cls, arpa_path, log=print):
        """Parse the ARPA file once, then reuse a pickle cache next to it (much faster start-up)."""
        arpa_path = Path(arpa_path)
        cache = arpa_path.with_name(arpa_path.name + ".cache.pkl")
        if cache.exists() and cache.stat().st_mtime >= arpa_path.stat().st_mtime:
            try:
                return cls.load(cache)
            except Exception:
                pass
        lm = cls.from_arpa(arpa_path, log=log)
        try:
            lm.save(cache)
        except Exception:
            pass
        return lm

    def add_hotwords(self, words, log10_score: float = -4.0):
        for w in words:
            w = w.upper().strip()
            if w and " " not in w:
                self.extra[w] = log10_score
                for k in range(1, len(w) + 1):
                    self.prefixes.add(w[:k])

    # ---------------------------------------------------------------- scoring
    def known(self, w):
        return w in self.prob or w in self.extra

    def log10p(self, ctx: tuple, w: str) -> float:
        """log10 P(w | ctx) with Katz-style back-off. ctx = up to order-1 previous words."""
        if w not in self.prob:
            return self.extra.get(w, UNK_LOG10)
        ctx = ctx[-(self.order - 1):] if self.order > 1 else ()
        back = 0.0
        for k in range(len(ctx), -1, -1):
            hist = ctx[len(ctx) - k:]
            key = " ".join(hist + (w,))
            p = self.prob.get(key)
            if p is not None:
                return p + back
            if k > 0:
                back += self.bow.get(" ".join(hist), 0.0)
        return UNK_LOG10


class Decoder:
    def __init__(self, vocab: list, lm: NgramLM | None, alpha: float = 0.5, beta: float = 1.0, beam: int = 32,
                 token_min_logp: float = math.log(1e-3), blank_skip: float = 0.999):
        self.vocab = list(vocab)
        assert self.vocab[1] == " ", "vocab[1] must be the space symbol"
        self.lm, self.alpha, self.beta, self.beam = lm, alpha, beta, beam
        self.token_min_logp, self.blank_skip_logp = token_min_logp, math.log(blank_skip)
        self.space = 1

    # LM part of a prefix's score, cached per prefix string (the LM only depends on the text)
    def _lm(self, pre: str, cache: dict):
        r = cache.get(pre)
        if r is not None:
            return r
        if not pre:
            r = (0.0, ("<s>",), 0.0)                    # committed LM score, LM context, partial-word penalty
        elif pre.endswith(" "):                          # "... W " -> the word W was just completed
            prev = pre[:-1]
            head = prev[:prev.rfind(" ") + 1]            # text before W ("" or ending with a space)
            w = prev[len(head):]
            base, ctx, _ = self._lm(head, cache)
            s = self.alpha * self.lm.log10p(ctx, w) * LN10 + self.beta
            r = (base + s, (ctx + (w,))[-max(1, self.lm.order - 1):], 0.0)
        else:                                            # "... par" -> partial word "par"
            head = pre[:pre.rfind(" ") + 1]
            part = pre[len(head):]
            base, ctx, _ = self._lm(head, cache)
            pen = 0.0 if part in self.lm.prefixes else self.alpha * UNK_LOG10 * LN10
            r = (base, ctx, pen)
        cache[pre] = r
        return r

    def _final_lm(self, pre: str, cache: dict) -> float:
        """LM score at the end of the audio: complete the last word, then </s>."""
        if self.lm is None:
            return 0.0
        p = pre.rstrip(" ")
        if not p:
            return 0.0
        full, ctx, _ = self._lm(p + " ", cache)
        return full + self.alpha * self.lm.log10p(ctx, "</s>") * LN10

    def decode(self, logp: np.ndarray) -> str:
        """logp: (T, V) natural-log CTC posteriors (log_softmax output)."""
        if self.lm is None:
            ids = logp.argmax(-1); out, prev = [], -1
            for k in ids:
                if k != prev and k != 0:
                    out.append(self.vocab[k])
                prev = k
            return " ".join("".join(out).split())
        V = logp.shape[1]
        cache = {}
        beams = {"": [0.0, NEG]}                         # prefix -> [log p ending in blank, log p ending in non-blank]
        cid = {c: i for i, c in enumerate(self.vocab)}
        for t in range(logp.shape[0]):
            row = logp[t]
            lpb = float(row[0])
            nxt = {}
            if lpb > self.blank_skip_logp:               # blank-dominated frame: no new prefixes
                for pre, (pb, pnb) in beams.items():
                    nb = _lse(pb, pnb) + lpb
                    nn_ = pnb + float(row[cid[pre[-1]]]) if pre else NEG
                    nxt[pre] = [nb, nn_]
            else:
                cand = [c for c in range(1, V) if row[c] > self.token_min_logp]
                for pre, (pb, pnb) in beams.items():
                    tot = _lse(pb, pnb)
                    e = nxt.setdefault(pre, [NEG, NEG])
                    e[0] = _lse(e[0], tot + lpb)
                    last = cid[pre[-1]] if pre else -1
                    if last >= 0:
                        e[1] = _lse(e[1], pnb + float(row[last]))
                    for c in cand:
                        lc = float(row[c])
                        if c == self.space and (not pre or pre.endswith(" ")):
                            e[0] = _lse(e[0], tot + lc)      # leading / double space: no new prefix
                            continue
                        new = pre + self.vocab[c]
                        ne = nxt.setdefault(new, [NEG, NEG])
                        ne[1] = _lse(ne[1], (pb if c == last else tot) + lc)
            # prune
            scored = []
            for pre, (pb, pnb) in nxt.items():
                base, _, pen = self._lm(pre, cache)
                scored.append((_lse(pb, pnb) + base + pen, pre))
            scored.sort(reverse=True)
            beams = {pre: nxt[pre] for _, pre in scored[:self.beam]}
        best, best_s = "", NEG
        for pre, (pb, pnb) in beams.items():
            s = _lse(pb, pnb) + self._final_lm(pre, cache)
            if s > best_s:
                best, best_s = pre, s
        return " ".join(best.split())
