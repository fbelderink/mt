import torch
from preprocessing.BPE import generate_bpe
from model.layers.linear import LinearLayer
from preprocessing.dictionary import Dictionary
from preprocessing.dataset import TranslationDataset


def test_linear_layer():
    layer = LinearLayer(3, 3, 5, bias=False)

    torch.manual_seed(0)
    x = torch.randn(3, 3, 1)

    print(layer(x))


def generate_dataset(multi30k_de, multi30k_en, window_size, num_operations, dict_de=None, dict_en=None, save_path=None):

    if not dict_de:
        german_ops = generate_bpe(multi30k_de, num_operations)
        dict_de = Dictionary(multi30k_de, german_ops)
        dict_de.save('eval/dict_de.pkl')

    if not dict_en:
        english_ops = generate_bpe(multi30k_en, num_operations)
        dict_en = Dictionary(multi30k_en, english_ops)
        dict_en.save('eval/dict_en.pkl')

    dataset = TranslationDataset(multi30k_de, multi30k_en, dict_de, dict_en, window_size)

    if save_path is not None:
        dataset.save(save_path)


def test_dataset_load(path):
    dataset = TranslationDataset.load(path)
    print(len(dataset))
    print(dataset[1])
