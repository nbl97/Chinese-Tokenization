import ahocorasick

class Dict:
	def __init__(self, dict_list, data_structure):
		if data_structure == "ac":
			self.A = ahocorasick.Automaton()
			for index, word in enumerate(dict_list):
			    self.A.add_word(word, (index, word))
			self.A.make_automaton()
			self.data_structure = "ac"
		if data_structure == "set":
			self.A = set(dict_list)
			self.data_structure = "set"
