from typing import List


# adds spaces between letters and 
# adds ␇ after the word to indicate the end of word 
def splitAndAddEndOfWordSymbol(word: str) -> str:
	res = ""
	for i in range(0,len(word)-1):
		res += word[i]

		# if current char and next char is a letter add a space
		if word[i] != " " and word[i+1] != " ":
			res += " " 

	# add last character
	res += word[len(word)-1]
	# add end of word symbol
	res += "␇"

	return res

# generates a dictionary of all token pairs in a given string and number of occurences 
def getAllTokenPairs(word: str) -> dict[str, int]:
	res = dict()
	subWordList = word.split()
	for i in range(0,len(subWordList)-1):
		current_join = subWordList[i]+" "+subWordList[i+1]
		if current_join in res:
			res[current_join] += 1
		else:
			res[current_join] = 1

	return res


# executes pulling-together-operation on a single word
def executeOperation(word: str, operation: List[str]) -> str:
	res = ""
	subWordList = word.split()

	for i in range(0,len(subWordList)-1):
		if subWordList[i] == operation[0] and subWordList[i+1] == operation[1]:
			res += subWordList[i]
		else:
			res += subWordList[i] + " "

	res += subWordList[-1] 
	return res

def generateBPE(training_data: str, operations_number: int) -> List[List[str]]:
	operations = []

	# step 1 
	# create a dictinary with split words and number of occurences 
	word_occurrences = dict()
	with open(training_data, 'r') as file:
		for line in file:
			for word in line.split():
				split_word = splitAndAddEndOfWordSymbol(word)

				if split_word in word_occurrences:
					word_occurrences[split_word] += 1
				else:
					word_occurrences[split_word] = 1 


	for i in range(operations_number):
		# step 2 
		# determine token pairs and count occurences using dictionary and save them into token_pairs 
		token_pairs = dict()
		for word in word_occurrences:
			# fetch all token pairs with number of occurrence for current word
			current_token_pair_dict = getAllTokenPairs(word)

			# update topen_pairs dictionary by transferring token pairs of current word 
			for current_token_pair in current_token_pair_dict:
					if current_token_pair in token_pairs:
						# add token pair's number of occurrence in current word multiplied by total number of occurence of 
						# the word the token pair is from to total count of occurrence of the token pair 
						token_pairs[current_token_pair] += current_token_pair_dict[current_token_pair]*word_occurrences[word]
					else:
						token_pairs[current_token_pair] = current_token_pair_dict[current_token_pair]*word_occurrences[word]


		# step 3
		# determine token pair with highest number of occurences and add to "operations" list 
		# if there are two tokens with same number of occurence, the one with higher lexicographic order is chosen
		if token_pairs: # check if dictionary is empty
			# TODO understand how that works
			highest_key = max(token_pairs, key=lambda k: (token_pairs[k], k))
			operations.append(highest_key.split())
		else:
			# nothing left to pull together; all subwords of every word have been pulled together 
			break

	
		# print(word_occurrences)
		# print()
		# print(operations)
		# print()


		# step 4
		# execute most recent pulling-together operation for every word in "word_occurrences" dictionary
		# create new dictionary that with updated keys
		updated_word_occurrences = dict()
		for word in word_occurrences:
			updated_word = executeOperation(word,operations[-1])
			updated_word_occurrences[updated_word] = word_occurrences[word]
		# replace old dictionary with new one
		word_occurrences = updated_word_occurrences
			
	return operations



def perfromBPEonText(operations: List[List[str]], source_file: str, target_file: str):
	# save the words with operations applied to them so that the operations don't have to be performed 
	# twice for the same word
	transformed_words = dict()
	with open(source_file, 'r') as s_file, open(target_file, "w") as t_file:
		for s_line in s_file:
			new_line = ""
			for word in s_line.split():
				if word in transformed_words:
					split_word = transformed_words[word]
				else:
					split_word = splitAndAddEndOfWordSymbol(word)
					for operation in operations:
						split_word = executeOperation(split_word, operation)

					transformed_words[word] = split_word

				new_line += split_word + " "

			# delte last space
			new_line = new_line[:-1]
			t_file.write(new_line+"\n")




def undoBPEonText(source_file: str, target_file: str):
	with open(source_file, 'r') as s_file, open(target_file, "w") as t_file:
		for s_line in s_file:
			new_line = s_line[0]
			for i in range(1,len(s_line)):
				if (s_line[i] == " " and s_line[i-1] != "␇") or s_line[i] == "␇":
					continue
				else:
					new_line += s_line[i]
			t_file.write(new_line)




