import torch
import numpy as np
import json

BOS, EOS = ' ', '\n'

with open('data/dictionary_char.json') as json_file:
    token_to_id = json.load(json_file)

with open('data/dictionary_category.json') as json_file:
    category = json.load(json_file)

def to_matrix(lines, max_len=None, pad=token_to_id[EOS], dtype=np.int64):
    max_len = max_len or max(map(len, lines))
    lines_ix = np.full([len(lines), max_len], pad, dtype=dtype)
    for i in range(len(lines)):
        line_ix = list(map(token_to_id.get, lines[i][:max_len]))
        lines_ix[i, :len(line_ix)] = line_ix
    return lines_ix

def predict(model, line):

    line = BOS + line.replace(EOS, ' ') + EOS
    line = [line.lower()]
    model.eval()
    matrix = torch.tensor(to_matrix(line))
    with torch.no_grad():
        pred = model(matrix)
        pred = torch.argmax(pred).cpu().detach().numpy()

    return {'result': category[str(float(pred))]}
