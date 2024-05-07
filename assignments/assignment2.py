from typing import List
from preprocessing.BPE import generate_bpe, perform_bpe
from preprocessing.dictionary import Dictionary
from preprocessing.batch import create_batch, get_index_batches
from utils.file_manipulation import save_batches


def task_evaluate(multi30k_de: List[List[str]], multi30k_en: List[List[str]]):
    joint_set = []
    joint_set.extend(multi30k_en)
    joint_set.extend(multi30k_de)

    for num_ops in [1000, 5000, 15000]:
        dic1 = Dictionary()
        dic1.generate_vocabulary(multi30k_de, generate_bpe(multi30k_de, num_ops))
        print(f"Number of unique words with {num_ops} operations on multi30k.de: {len(dic1)}")

        dic2 = Dictionary()
        dic2.generate_vocabulary(multi30k_en, generate_bpe(multi30k_en, num_ops))
        print(f"Number of unique words with {num_ops} operations on multi30k.en: {len(dic2)}")

        dic3 = Dictionary()
        dic3.generate_vocabulary(joint_set, generate_bpe(joint_set, num_ops))
        print(f"Number of unique words with {num_ops} operations on multi30k.de joint with multi30k.en: {len(dic3)}")

        print()


def task_batches(window_size, batch_size, multi30k_de, multi30k_en):
    num_operations = 1000

    german_ops = generate_bpe(multi30k_de, num_operations)
    english_ops = generate_bpe(multi30k_en, num_operations)

    bpe_de, transformed_words_de = perform_bpe(multi30k_de, german_ops)
    bpe_en, transformed_words_en = perform_bpe(multi30k_en, english_ops)

    batches = create_batch(bpe_de[1100:1200],
                           bpe_en[1100:1200],
                           window_size,
                           batch_size)

    save_batches('eval/string_batches', batches)

    dic_de = Dictionary()
    dic_de.generate_vocabulary([], german_ops, transformed_words=transformed_words_de)

    dic_en = Dictionary()
    dic_en.generate_vocabulary([], english_ops, transformed_words=transformed_words_en)

    index_batches = get_index_batches(batches, dic_de, dic_en)
    save_batches('eval/index_batches', index_batches)
