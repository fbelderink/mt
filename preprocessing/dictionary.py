from BPE import *

class Dictionary:
	# Initialize the dictionary
	def __init__(self):
		self.vocabulary = dict() 

	def __str__(self):
		# Return a string representation of the dictionary's contents
		return str(list(self.vocabulary.items()))

	def getStringAtIndex(self, index: int):
		# Retrieve the string at a given index in the dictionary
		return self.vocabulary.get(index, "Index not found")

	def setStringAtIndex(self, index: int, value: str):
		# Set or update the string at a specific index in the dictionary
		self.vocabulary[index] = value

	def getIndexOfString(self, string: str):
		# Get the index of a given string in the dictionary
		for key, val in self.vocabulary.items():
			if val == string:
				return key
		return None  # If not found, return None

	def addString(self, value: str):
		# Find the first available index to add the specified string
		index = 0
		# Loop to find the first available index
		while index in self.vocabulary:
			index += 1
		self.vocabulary[index] = value
		return index  # Return the index where the string was added

	def isContained(self, item):
		# Check if a specific index or string is in the dictionary
		if isinstance(item, int):
			return item in self.vocabulary
		elif isinstance(item, str):
			return item in self.vocabulary.values()
		else:
			raise TypeError("Invalid type. Expected int or str.")

	def emptyDictionary(self):
		# Clear the dictionary's contents
		self.vocabulary.clear()


	def generateVocabulary(self,source_file: str, operations: List[List[str]]):
		# save the words with operations applied to them so that the 
		# operations don't have to be performed twice for the same word
		transformed_words = dict()
		
		with open(source_file, 'r') as s_file:
			for s_line in s_file:
				for word in s_line.split():
					if not (word in transformed_words):
						split_word = splitAndAddEndOfWordSymbol(word)
						for operation in operations:
							split_word = executeOperation(split_word, operation)

						transformed_words[word] = split_word

		self.emptyDictionary()
		for split_word in transformed_words.values():
			for subword in split_word.split():
				if not self.isContained(subword):
					self.addString(subword)


	def applyVocabularyToText(self, source_file: str, target_file: str):
		with open(source_file, 'r') as s_file, open(target_file, "w") as t_file:
			for s_line in s_file:
				new_line = ""
				for word in s_line.split():
					if self.isContained(word):
						new_line += word + " "
					else:
						new_line += "<UNK> "
				# delte last space
				new_line = new_line[:-1]
				t_file.write(new_line+"\n")















