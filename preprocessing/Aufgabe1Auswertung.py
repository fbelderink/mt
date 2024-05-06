from dictionary import *
from BPE import *


dic1 = Dictionary()
dic1.generateVocabulary(["multi30k.de"],generateBPE(["multi30k.de"],1000))
print("Number unique words with 1000 operations on multi30k.de: " + str(dic1.getSize()))

dic2 = Dictionary()
dic2.generateVocabulary(["multi30k.en"],generateBPE(["multi30k.en"],1000))
print("Number unique words with 1000 operations on multi30k.en: " + str(dic2.getSize()))

dic3 = Dictionary()
dic3.generateVocabulary(["multi30k.en","multi30k.de"],generateBPE(["multi30k.en","multi30k.de"],1000))
print("Number unique words with 1000 operations on multi30k.de joint with multi30k.en: " + str(dic3.getSize()))

print()

dic4 = Dictionary()
dic4.generateVocabulary(["multi30k.de"],generateBPE(["multi30k.de"],5000))
print("Number unique words with 5000 operations on multi30k.de: " + str(dic4.getSize()))

dic5 = Dictionary()
dic5.generateVocabulary(["multi30k.en"],generateBPE(["multi30k.en"],5000))
print("Number unique words with 5000 operations on multi30k.en: " + str(dic5.getSize()))

dic6 = Dictionary()
dic6.generateVocabulary(["multi30k.en","multi30k.de"],generateBPE(["multi30k.en","multi30k.de"],5000))
print("Number unique words with 5000 operations on multi30k.de joint with multi30k.en: " + str(dic6.getSize()))

print()

dic7 = Dictionary()
dic7.generateVocabulary(["multi30k.de"],generateBPE(["multi30k.de"],15000))
print("Number unique words with 15000 operations on multi30k.de: " + str(dic7.getSize()))

dic8 = Dictionary()
dic8.generateVocabulary(["multi30k.en"],generateBPE(["multi30k.en"],15000))
print("Number unique words with 15000 operations on multi30k.en: " + str(dic8.getSize()))

dic9 = Dictionary()
dic9.generateVocabulary(["multi30k.en","multi30k.de"],generateBPE(["multi30k.en","multi30k.de"],15000))
print("Number unique words with 15000 operations on multi30k.de joint with multi30k.en: " + str(dic9.getSize()))




