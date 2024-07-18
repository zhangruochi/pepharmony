import os
import sys
import mlflow
import torch
from omegaconf import OmegaConf
from model.esm.data import Alphabet


def load_model(model_path):
    sys.path.insert(0, os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model_peptide = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model = model_peptide.model.model_seq
    model.eval()
    return model


class Inference(object):

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def extract_features(self, input_seq):

        protein_input = [("protein_{}".format(i), seq)
                         for i, seq in enumerate(input_seq)]
        _, _, protein_input = self.tokenizer(protein_input)
        protein_input = protein_input.to(self.device)
        with torch.no_grad():
            embeds = self.model(None, protein_input)['residue_feature']
        embeds = embeds.cpu().numpy()

        assert len(embeds) == len(input_seq)

        return embeds


if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/inference.yaml")
    device = torch.device(cfg.device)
    model_path = cfg.model.pepharmony.model_path
    model = load_model(model_path)
    tokenizer = Alphabet.from_architecture(
        cfg.token_arch).get_batch_converter()
    model.to(device)

    ## Get the sequence list
    instances = {
        "instances": {
            "sequences": [
                'HPRLSQYKSKYSSLEQSERRRRLLELQKSKRLDYVNHARR',
                'SMWTEHKSPDGRTYYYNTETKQSTWEKP', 'ACGSCRKKCKGSGKCINGRCKCY',
                'GDCHKFLGWCRGEKDPCCEHLTCHVKHGWCVWDGTI',
                'TVCNLRRCQLSCRSLGLLGKCIGVKCECVKH',
                'EDIDECQELPGLCQGGKCINTFGSFQCRCPTGYYLNEDTRVCD'
            ]
        }
    }
    input_seq = instances["instances"]["sequences"]

    print("total sequences: {}".format(len(input_seq)))

    feature_extractor = Inference(tokenizer, model, device)
    embeds = feature_extractor.extract_features(input_seq)

    print(embeds)
