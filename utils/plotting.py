import torch
import numpy as np

from torch.utils.data import Subset
import torch.nn.functional as F


def tokens2str(tokens):
    pretok_sent = ""
    for tok in tokens:
        if tok.startswith("##"):
            pretok_sent += tok[2:]
        else:
            pretok_sent += "" + tok

    return pretok_sent


def print_prediction(net, dataset, device):
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    label_map = {i: label for i, label in enumerate(dataset.label_list, 1)}
    tokenizer = dataset.tokenizer

    idx = np.random.randint(len(dataset))
    net.eval()

    # {
    #     1: 'O',
    #     2: 'B-iupac',
    #     3: 'I-iupac',
    #     4: 'B-prefix',
    #     5: 'I-prefix',
    #     6: '[CLS]',
    #     7: '[SEP]'
    # }

    with torch.no_grad():
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = dataset[
            idx]

        ids = input_ids.cpu().numpy()[l_mask.cpu().numpy() == 1].tolist()
        raw_str = tokens2str(tokenizer.convert_ids_to_tokens(ids))

        input_ids = input_ids.unsqueeze(0).to(device)
        input_mask = input_mask.unsqueeze(0).to(device)
        segment_ids = segment_ids.unsqueeze(0).to(device)
        valid_ids = valid_ids.unsqueeze(0).to(device)
        logits = net(input_ids=input_ids,
                          token_type_ids=segment_ids,
                          attention_mask=input_mask,
                          valid_ids=valid_ids)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        pre_ids = logits.detach().cpu().numpy()[0]

        true_tokens = []
        predicted_tokens = []

        for j, m in enumerate(label_ids):
            true_tokens.append(label_map.get(label_ids[j].item(), "PAD"))
            predicted_tokens.append(label_map.get(pre_ids[j], "PAD"))

            if label_ids[j] == len(label_map):
                break

    title = "Text: {}\n\nTrue: {}\n\nPred: {}".format(raw_str, true_tokens, predicted_tokens)

    return title
