"""Build calibration and held-out evaluation sequences (token ids) for inkling2_mlx.prune.

A mix of chat question answering, worked maths with a reasoning channel, code and encyclopedic prose,
rendered with the model's own chat template and cut to one common length.
"""
import json, sys, urllib.parse, urllib.request
sys.path.insert(0, ".")
from inkling2_mlx.processing import Processor

TOKENIZER_DIR, OUT, T = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 512


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    req = urllib.request.Request("https://datasets-server.huggingface.co/rows?" + q, headers={"User-Agent": "inkling2-mlx-calibration"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [x["row"] for x in json.load(r)["rows"]]


alpaca = rows("tatsu-lab/alpaca", "default", "train", 0, 100) + rows("tatsu-lab/alpaca", "default", "train", 5000, 100)
gsm = rows("openai/gsm8k", "main", "train", 0, 100)
mbpp = rows("google-research-datasets/mbpp", "full", "train", 0, 100)
wiki = [r["text"].strip() for r in rows("Salesforce/wikitext", "wikitext-2-raw-v1", "train", 0, 100) + rows("Salesforce/wikitext", "wikitext-2-raw-v1", "train", 100, 100)]
wiki = [t for t in wiki if len(t.split()) > 120 and not t.startswith("=")]
print(f"fetched: {len(alpaca)} chat, {len(gsm)} maths, {len(mbpp)} code, {len(wiki)} prose paragraphs")


def chat_turns(items):
    out = []
    for r in items:
        prompt = r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")
        out += [{"role": "user", "content": prompt}, {"role": "assistant", "content": r["output"]}]
    return out


def maths_turns(items):
    out = []
    for r in items:
        work, _, answer = r["answer"].partition("####")
        out += [{"role": "user", "content": r["question"]},
                {"role": "assistant", "reasoning_content": work.strip(), "content": f"The answer is {answer.strip()}."}]
    return out


def code_turns(items):
    out = []
    for r in items:
        out += [{"role": "user", "content": r["text"]}, {"role": "assistant", "content": "```python\n" + r["code"].strip() + "\n```"}]
    return out


def prose_turns(items):
    out = []
    for text in items:
        words = text.split()
        out += [{"role": "user", "content": "Continue this passage in the same style:\n\n" + " ".join(words[:60])},
                {"role": "assistant", "content": " ".join(words[60:])}]
    return out


processor = Processor(TOKENIZER_DIR)


def sequences(turns, count, effort, lead=None):
    """Pack consecutive exchanges into conversations of at least T tokens, cut to exactly T."""
    out, cursor = [], 0
    while len(out) < count:
        messages = list(lead or [])
        while True:
            messages += turns[cursor : cursor + 2]
            cursor += 2
            if cursor >= len(turns):
                raise SystemExit("not enough source text")
            ids = processor(messages, reasoning_effort=effort)["input_ids"]
            if len(ids) >= T:
                break
        out.append(ids[:T])
    return out, cursor


probe = [{"role": "user", "content": "List the capital cities of France, Germany, Italy, Spain and Japan, in that order, "
          "separated by commas. Reply with the five names only."},
         {"role": "assistant", "content": "Paris, Berlin, Rome, Madrid, Tokyo"}]
probe_start = len(processor(probe[:1], reasoning_effort="none")["input_ids"])
probe_end = len(processor(probe, reasoning_effort="none")["input_ids"])

chat, maths, code, prose = chat_turns(alpaca), maths_turns(gsm), code_turns(mbpp), prose_turns(wiki)
calibration, evaluation, labels = [], [], []
c1, used = sequences(chat, 5, "none"); calibration += c1
c2, used2 = sequences(chat[used:], 5, None); calibration += c2
m1, mused = sequences(maths, 6, None); calibration += m1
k1, kused = sequences(code, 4, "none"); calibration += k1
p1, pused = sequences(prose, 4, "none"); calibration += p1
rest = chat[used + used2 :]
e, eu = sequences(rest, 1, "none", lead=probe); evaluation += e; labels.append("chat (starts with the capitals probe)")
e, _ = sequences(rest[eu:], 1, None); evaluation += e; labels.append("chat, default reasoning effort")
e, _ = sequences(maths[mused:], 1, None); evaluation += e; labels.append("maths with reasoning")
e, _ = sequences(code[kused:], 1, "none"); evaluation += e; labels.append("code")
e, _ = sequences(prose[pused:], 1, "none"); evaluation += e; labels.append("prose continuation")
assert all(len(s) == T for s in calibration + evaluation)
json.dump({"calibration": calibration, "evaluation": evaluation, "evaluation_labels": labels,
           "probe": {"row": 0, "start": probe_start, "end": probe_end}}, open(OUT, "w"))
print(f"{len(calibration)} calibration x {T} = {len(calibration) * T} tokens; {len(evaluation)} evaluation sequences: {labels}")
print("probe answer tokens:", [processor.decode([t]) for t in evaluation[0][probe_start:probe_end]])
print("sample:", repr(processor.decode(calibration[11][:90])))
