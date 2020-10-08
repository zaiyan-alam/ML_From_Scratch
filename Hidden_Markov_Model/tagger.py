import numpy as np
from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):

	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	S = len(tags)
    # Find state symbols
	state_dict = {}
	for i in range(S):
		state_dict[tags[i]] = i
    # Find initial probability
    sentences = len(train_data)
    pi = [0.0]*S
    for sen in range(sentences):
        ind = state_dict[train_data[sen].tags[0]]
        pi[ind] = pi[ind] + 1
    for x in range(S):
        pi[x] = pi[x]/sentences
    # Find obs_dict
    obs_dict = {}
    ind = 0
    for sen in range(sentences):
        sentence = train_data[sen].words
        for word in sentence:
            if word not in obs_dict.keys():
                obs_dict[word] = ind
                ind = ind +1

    # Find transition probabilities - A
    A = np.zeros([S, S])
    start = [0.0]*S
    for sen in range(sentences):
        sentence = train_data[sen].tags
        for i in range(len(sentence)-1):
            s = sentence[i]
            start[state_dict[s]] +=1
            sp = sentence[i+1]
            A[state_dict[s]][state_dict[sp]] += 1
    for i in range(S):
        if start[i] != 0:
            A[i] = [x/start[i] for x in A[i]]

    # Find emission probabilities - B
    B = np.zeros([S, ind])
    for sen in range(sentences):
        start[state_dict[train_data[sen].tags[-1]]] += 1
        for i in range(len(train_data[sen].tags)):
            s = train_data[sen].tags[i]
            o = train_data[sen].words[i]
            B[state_dict[s]][obs_dict[o]] += 1
    for i in range(S):
        if start[i] != 0:
            B[i] = [x/start[i] for x in B[i]]

    model = HMM(pi, A, B, obs_dict, state_dict)
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	L = len(test_data)
    S = len(tags)

    # updates for new words
    ind = max(model.obs_dict.values()) + 1
    z = np.full((S, 1), 1e-6)

    # Calling viterbi algorithm
    for i in range(L):
        for word in test_data[i].words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = ind
                model.B = np.append(model.B, z, axis=1)
                ind += 1

        tagging.append(model.viterbi(test_data[i].words))
	return tagging
