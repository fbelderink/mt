from dictionary import *
from BPE import *
from utils.file_manipulation import *
from preprocessing.batch import *


def eval_task1():
    dic1 = Dictionary()
    dic1.generateVocabulary(["../data/multi30k.de"], generateBPE(["../data/multi30k.de"], 1000))
    print("Number unique words with 1000 operations on multi30k.de: " + str(dic1.getSize()))

    dic2 = Dictionary()
    dic2.generateVocabulary(["../data/multi30k.en"], generateBPE(["../data/multi30k.en"], 1000))
    print("Number unique words with 1000 operations on multi30k.en: " + str(dic2.getSize()))

    dic3 = Dictionary()
    dic3.generateVocabulary(["../data/multi30k.en", "../data/multi30k.de"],
                            generateBPE(["../data/multi30k.en", "../data/multi30k.de"], 1000))
    print("Number unique words with 1000 operations on multi30k.de joint with multi30k.en: " + str(dic3.getSize()))

    print()

    dic4 = Dictionary()
    dic4.generateVocabulary(["../data/multi30k.de"], generateBPE(["../data/multi30k.de"], 5000))
    print("Number unique words with 5000 operations on multi30k.de: " + str(dic4.getSize()))

    dic5 = Dictionary()
    dic5.generateVocabulary(["../data/multi30k.en"], generateBPE(["../data/multi30k.en"], 5000))
    print("Number unique words with 5000 operations on multi30k.en: " + str(dic5.getSize()))

    dic6 = Dictionary()
    dic6.generateVocabulary(["../data/multi30k.en", "../data/multi30k.de"],
                            generateBPE(["../data/multi30k.en", "../data/multi30k.de"], 5000))
    print("Number unique words with 5000 operations on multi30k.de joint with multi30k.en: " + str(dic6.getSize()))

    print()

    dic7 = Dictionary()
    dic7.generateVocabulary(["../data/multi30k.de"], generateBPE(["../data/multi30k.de"], 15000))
    print("Number unique words with 15000 operations on multi30k.de: " + str(dic7.getSize()))

    dic8 = Dictionary()
    dic8.generateVocabulary(["../data/multi30k.en"], generateBPE(["../data/multi30k.en"], 15000))
    print("Number unique words with 15000 operations on multi30k.en: " + str(dic8.getSize()))

    dic9 = Dictionary()
    dic9.generateVocabulary(["../data/multi30k.en", "../data/multi30k.de"],
                            generateBPE(["../data/multi30k.en", "../data/multi30k.de"], 15000))
    print("Number unique words with 15000 operations on multi30k.de joint with multi30k.en: " + str(dic9.getSize()))


def eval_batches():
    num_operations = 1000

    dic1 = Dictionary()
    dic1.generateVocabulary(["../data/multi30k.de"], generateBPE(["../data/multi30k.de"], num_operations))

    print(dic1)

    dic2 = Dictionary()
    dic2.generateVocabulary(["../data/multi30k.en"], generateBPE(["../data/multi30k.en"], num_operations))

    print(dic2)

    multi30k_de = load_data('../data/multi30k.de')
    multi30k_en = load_data('../data/multi30k.en')

    batches = create_batch(multi30k_de, multi30k_en, 2, 200)

    save_batches('../eval/string_batches', batches)
    index_batches = get_index_batches(batches, dic1, dic2)

    save_batches('../eval/index_batches', index_batches)


if __name__ == '__main__':
    eval_batches()
    #eval_task1()
