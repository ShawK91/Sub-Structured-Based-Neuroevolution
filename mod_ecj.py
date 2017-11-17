from random import randint
import fastrand
import math, copy
import numpy as np
import random
from scipy.special import expit
import sys,os, cPickle

class normal_net:
    def __init__(self, num_input, num_hnodes, num_output, mean = 0, std = 1):
        self.num_substructures = 4
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.net_output = [] * num_output
        #W_01 matrix contains the weigh going from input (0) to the 1st hidden layer(1)
        self.w_01 = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1))) #Initilize weight using Gaussian distribution
        self.w_01 = np.mat(np.reshape(self.w_01, (num_hnodes, (num_input + 1)))) #Reshape the array to the weight matrix
        self.w_12 = np.mat(np.random.normal(mean, std, num_output*(num_hnodes + 1)))
        self.w_12 = np.mat(np.reshape(self.w_12, (num_output, (num_hnodes + 1)))) #Reshape the array to the weight matrix

    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def sigmoid(self, layer_input): #Sigmoid transform

        #Just make sure the sigmoid does not explode and cause a math error
        p = layer_input.A[0][0]
        if p > 700:
            ans = 0.999999
        elif p < -700:
            ans = 0.000001
        else:
            ans =  1 / (1 + math.exp(-p))

        return ans

    def fast_sigmoid(self, layer_input):  # Sigmoid transform
        layer_input = expit(layer_input)
        # for i in layer_input: i[0] = i / (1 + math.fabs(i))
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def format_input(self, input, add_bias = True): #Formats and adds bias term to given input at the end
        if add_bias:
            input = np.concatenate((input, [1.0]))
        return np.mat(input)

    def format_mat(self, input):
        ig = np.mat([1])
        return np.concatenate((input, ig))

    def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
        #First hidden layer
        self.input = self.format_input(input) #Format and add bias term at the end
        z1 = self.linear_combination(self.w_01, self.input.transpose()) #Forward pass linear
        z1 = self.format_input(z1, False)#Format
        h1 = self.fast_sigmoid(z1) #Use fast sigmoid transform

        #Output layer
        #h1 = np.vstack((h1,[1])) #Add bias term
        h1 = self.format_mat(h1)
        z2 = self.w_12 * h1 #Forward pass linear
        self.net_output = (self.fast_sigmoid((z2))) #Use sigmoid transform
        return np.array(self.net_output).tolist()

    def get_weights(self):
        w1 = np.array(self.w_01).flatten().copy()
        w2 = np.array(self.w_12).flatten().copy()
        weights = np.concatenate((w1, w2 ))
        return weights

    def set_weights(self, weights):
        w1 = weights[:self.num_hnodes*(self.num_input + 1)]
        w2 = weights[self.num_hnodes*(self.num_input + 1):]
        self.w_01 = np.mat(np.reshape(w1, (self.num_hnodes, (self.num_input + 1)))) #Reshape the array to the weight matrix
        self.w_12 = np.mat(np.reshape(w2, (self.num_output, (self.num_hnodes + 1)))) #Reshape the array to the weight matrix

    def reset_bank(self):
        k = 1

    def set_bank(self):
        k = 1

class GRUMB:
    def __init__(self, params):

        mean = 0; std = 1
        num_input = params.num_input; num_output = params.num_output; num_hnodes = params.num_hnodes

        #Adaptive components (plastic with network running)
        self.last_output = np.mat(np.zeros(num_output)).transpose()
        self.memory_cell = np.mat(np.random.normal(mean, std, num_hnodes)).transpose() #Memory Cell

        #Banks for adaptive components, that can be used to reset
        #self.bank_last_output = self.last_output[:]
        self.bank_memory_cell = np.copy(self.memory_cell) #Memory Cell

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inpgate = np.mat(np.reshape(self.w_inpgate, (num_hnodes, (num_input + 1))))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inpgate = np.mat(np.reshape(self.w_rec_inpgate, (num_hnodes, (num_output + 1))))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_inpgate = np.mat(np.reshape(self.w_mem_inpgate, (num_hnodes, (num_hnodes + 1))))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inp = np.mat(np.reshape(self.w_inp, (num_hnodes, (num_input + 1))))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inp = np.mat(np.reshape(self.w_rec_inp, (num_hnodes, (num_output + 1))))

        #read gate
        self.w_readgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_readgate = np.mat(np.reshape(self.w_readgate, (num_hnodes, (num_input + 1))))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_readgate = np.mat(np.reshape(self.w_rec_readgate, (num_hnodes, (num_output + 1))))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_readgate = np.mat(np.reshape(self.w_mem_readgate, (num_hnodes, (num_hnodes + 1))))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_writegate = np.mat(np.reshape(self.w_writegate, (num_hnodes, (num_input + 1))))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_writegate = np.mat(np.reshape(self.w_rec_writegate, (num_hnodes, (num_output + 1))))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_writegate = np.mat(np.reshape(self.w_mem_writegate, (num_hnodes, (num_hnodes + 1))))

        #Output weights
        self.w_output = np.mat(np.random.normal(mean, std, num_output*(num_hnodes + 1)))
        self.w_output = np.mat(np.reshape(self.w_output, (num_output, (num_hnodes + 1)))) #Reshape the array to the weight matrix

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_hid_out': self.w_output}

    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def sigmoid(self, layer_input): #Sigmoid transform

        #Just make sure the sigmoid does not explode and cause a math error
        p = layer_input.A[0][0]
        if p > 700:
            ans = 0.999999
        elif p < -700:
            ans = 0.000001
        else:
            ans =  1 / (1 + math.exp(-p))

        return ans

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        layer_input = expit(layer_input)
        #for i in layer_input: i[0] = i / (1 + math.fabs(i))
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def format_input(self, input, add_bias = True): #Formats and adds bias term to given input at the end
        if add_bias:
            input = np.concatenate((input, [1.0]))
        return np.mat(input)

    def format_memory(self, memory):
        ig = np.mat([1])
        return np.concatenate((memory, ig))

    #Memory_write gate
    def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
        self.input = self.format_input(input).transpose()  # Format and add bias term at the end
        last_memory = self.format_memory(self.memory_cell)
        last_output = self.format_memory(self.last_output)

        #Input gate
        ig_1 = self.linear_combination(self.w_inpgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_inpgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_inpgate, last_memory)
        input_gate_out = ig_1 + ig_2 + ig_3
        input_gate_out = self.fast_sigmoid(input_gate_out)

        #Input processing
        ig_1 = self.linear_combination(self.w_inp, self.input)
        ig_2 = self.linear_combination(self.w_rec_inp, last_output)
        block_input_out = ig_1 + ig_2
        block_input_out = self.fast_sigmoid(block_input_out)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #read Gate
        ig_1 = self.linear_combination(self.w_readgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_readgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_readgate, last_memory)
        read_gate_out = ig_1 + ig_2 + ig_3
        read_gate_out = self.fast_sigmoid(read_gate_out)

        #Memory Output
        memory_output = np.multiply(read_gate_out, self.memory_cell)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = memory_output + input_out

        #Write gate (memory cell)
        ig_1 = self.linear_combination(self.w_writegate, self.input)
        ig_2 = self.linear_combination(self.w_rec_writegate, last_output)
        ig_3 = self.linear_combination(self.w_mem_writegate, last_memory)
        write_gate_out = ig_1 + ig_2 + ig_3
        write_gate_out = self.fast_sigmoid(write_gate_out)

        #Write to memory Cell - Update memory
        self.memory_cell += np.multiply(write_gate_out, np.tanh(hidden_act))
        #temp = np.multiply(write_gate_out, hidden_act)
        #temp = self.fast_sigmoid(temp)
        #self.memory_cell += temp

        #Compute final output
        hidden_act = self.format_memory(hidden_act)
        self.last_output = self.linear_combination(self.w_output, hidden_act)
        self.last_output = self.fast_sigmoid(self.last_output)
        #print self.last_output
        return np.array(self.last_output).tolist()

    def reset(self):
        self.last_output *= 0  # last output
        self.memory_cell *= 0

    def get_weights(self):
        #TODO NOT OPERATIONAL
        w1 = np.array(self.w_01).flatten().copy()
        w2 = np.array(self.w_12).flatten().copy()
        weights = np.concatenate((w1, w2 ))
        return weights

    def set_weights(self, weights):
        #Input gates
        start = 0; end = self.num_hnodes*(self.num_input + 1)
        w_inpgate = weights[start:end]
        self.w_inpgate = np.mat(np.reshape(w_inpgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inpgate = weights[start:end]
        self.w_rec_inpgate = np.mat(np.reshape(w_rec_inpgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_inpgate = weights[start:end]
        self.w_mem_inpgate = np.mat(np.reshape(w_mem_inpgate, (self.num_hnodes, (self.num_hnodes + 1))))

        # Block Inputs
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_inp = weights[start:end]
        self.w_inp = np.mat(np.reshape(w_inp, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inp = weights[start:end]
        self.w_rec_inp = np.mat(np.reshape(w_rec_inp, (self.num_hnodes, (self.num_output + 1))))

        #read Gates
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_readgate = weights[start:end]
        self.w_readgate = np.mat(np.reshape(w_readgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_readgate = weights[start:end]
        self.w_rec_readgate = np.mat(np.reshape(w_rec_readgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_readgate = weights[start:end]
        self.w_mem_readgate = np.mat(np.reshape(w_mem_readgate, (self.num_hnodes, (self.num_hnodes + 1))))

        #Output weights
        start = end; end += self.num_output*(self.num_hnodes + 1)
        w_output= weights[start:end]
        self.w_output = np.mat(np.reshape(w_output, (self.num_output, (self.num_hnodes + 1))))

        #Memory Cell (prior)
        start = end; end += self.num_hnodes
        memory_cell= weights[start:end]
        self.memory_cell = np.mat(memory_cell).transpose()

class SSNE:
    def __init__(self, parameters):
        self.parameters = parameters;
        self.num_elitists = int(parameters.elite_fraction * parameters.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_substructures = 13

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):

        # INPUT GATES
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                gene1.w_inpgate[ind_cr, :] = gene2.w_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                gene2.w_inpgate[ind_cr, :] = gene1.w_inpgate[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                gene1.w_rec_inpgate[ind_cr, :] = gene2.w_rec_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                gene2.w_rec_inpgate[ind_cr, :] = gene1.w_rec_inpgate[ind_cr, :]
            else:
                continue

        # Layer 3
        num_cross_overs = randint(1, len(gene1.w_mem_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                gene1.w_mem_inpgate[ind_cr, :] = gene2.w_mem_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                gene2.w_mem_inpgate[ind_cr, :] = gene1.w_mem_inpgate[ind_cr, :]
            else:
                continue

        # BLOCK INPUTS
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_inp))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_inp) - 1)
                gene1.w_inp[ind_cr, :] = gene2.w_inp[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_inp) - 1)
                gene2.w_inp[ind_cr, :] = gene1.w_inp[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_inp))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                gene1.w_rec_inp[ind_cr, :] = gene2.w_rec_inp[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                gene2.w_rec_inp[ind_cr, :] = gene1.w_rec_inp[ind_cr, :]
            else:
                continue

        # read GATES
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_readgate) - 1)
                gene1.w_readgate[ind_cr, :] = gene2.w_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_readgate) - 1)
                gene2.w_readgate[ind_cr, :] = gene1.w_readgate[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_readgate) - 1)
                gene1.w_rec_readgate[ind_cr, :] = gene2.w_rec_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_readgate) - 1)
                gene2.w_rec_readgate[ind_cr, :] = gene1.w_rec_readgate[ind_cr, :]
            else:
                continue

        # Layer 3
        num_cross_overs = randint(1, len(gene1.w_mem_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_mem_readgate) - 1)
                gene1.w_mem_readgate[ind_cr, :] = gene2.w_mem_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_mem_readgate) - 1)
                gene2.w_mem_readgate[ind_cr, :] = gene1.w_mem_readgate[ind_cr, :]
            else:
                continue

        # OUTPUT WEIGHTS
        num_cross_overs = randint(1, len(gene1.w_output))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_output) - 1)
                gene1.w_output[ind_cr, :] = gene2.w_output[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_output) - 1)
                gene2.w_output[ind_cr, :] = gene1.w_output[ind_cr, :]
            else:
                continue

        # MEMORY CELL (PRIOR)
        # 1-dimensional so point crossovers
        num_cross_overs = randint(1, int(gene1.w_rec_readgate.shape[1] / 2))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, gene1.w_rec_readgate.shape[1] - 1)
                gene1.w_rec_readgate[0, ind_cr:] = gene2.w_rec_readgate[0, ind_cr:]
            elif rand < 0.66:
                ind_cr = randint(0, gene1.w_rec_readgate.shape[1] - 1)
                gene2.w_rec_readgate[0, :ind_cr] = gene1.w_rec_readgate[0, :ind_cr]
            else:
                continue

        if self.num_substructures == 13:  # Only for NTM
            # WRITE GATES
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_writegate) - 1)
                    gene1.w_writegate[ind_cr, :] = gene2.w_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_writegate) - 1)
                    gene2.w_writegate[ind_cr, :] = gene1.w_writegate[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_rec_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                    gene1.w_rec_writegate[ind_cr, :] = gene2.w_rec_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                    gene2.w_rec_writegate[ind_cr, :] = gene1.w_rec_writegate[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_mem_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                    gene1.w_mem_writegate[ind_cr, :] = gene2.w_mem_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                    gene2.w_mem_writegate[ind_cr, :] = gene1.w_mem_writegate[ind_cr, :]
                else:
                    continue

    def regularize_weight(self, weight):
        if weight > self.parameters.weight_magnitude_limit:
            weight = self.parameters.weight_magnitude_limit
        if weight < -self.parameters.weight_magnitude_limit:
            weight = -self.parameters.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.0

        # Initiate distribution
        if self.parameters.mut_distribution == 1:  # Gaussian
            ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
        elif self.parameters.mut_distribution == 2:  # Laplace
            ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
        elif self.parameters.mut_distribution == 3:  # Uniform
            ss_mut_dist = np.random.uniform(0, 1, self.num_substructures)
        else:
            ss_mut_dist = [1] * self.num_substructures


        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        for ssne_prob, key in zip(ss_mut_dist, keys): #For each structure
            if random.random()<ssne_prob:

                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                      W[key][
                                                                                          ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                                                                                          ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                        W[key][ind_dim1, ind_dim2])

    def epoch(self, pop, fitnesses):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitnesses);
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.parameters.pop_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning eleitst candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            pop[i] = copy.deepcopy(pop[off_i])
            pop[j] = copy.deepcopy(pop[off_j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.parameters.pop_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.mutation_prob:
                    self.mutate_inplace(pop[i])

class Evo:
    def __init__(self, parameters):
        self.parameters = parameters;
        self.num_elitists = int(parameters.elite_fraction * parameters.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_substructures = 13

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):

        # INPUT GATES
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                gene1.w_inpgate[ind_cr, :] = gene2.w_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                gene2.w_inpgate[ind_cr, :] = gene1.w_inpgate[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                gene1.w_rec_inpgate[ind_cr, :] = gene2.w_rec_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                gene2.w_rec_inpgate[ind_cr, :] = gene1.w_rec_inpgate[ind_cr, :]
            else:
                continue

        # Layer 3
        num_cross_overs = randint(1, len(gene1.w_mem_inpgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                gene1.w_mem_inpgate[ind_cr, :] = gene2.w_mem_inpgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                gene2.w_mem_inpgate[ind_cr, :] = gene1.w_mem_inpgate[ind_cr, :]
            else:
                continue

        # BLOCK INPUTS
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_inp))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_inp) - 1)
                gene1.w_inp[ind_cr, :] = gene2.w_inp[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_inp) - 1)
                gene2.w_inp[ind_cr, :] = gene1.w_inp[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_inp))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                gene1.w_rec_inp[ind_cr, :] = gene2.w_rec_inp[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                gene2.w_rec_inp[ind_cr, :] = gene1.w_rec_inp[ind_cr, :]
            else:
                continue

        # read GATES
        # Layer 1
        num_cross_overs = randint(1, len(gene1.w_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_readgate) - 1)
                gene1.w_readgate[ind_cr, :] = gene2.w_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_readgate) - 1)
                gene2.w_readgate[ind_cr, :] = gene1.w_readgate[ind_cr, :]
            else:
                continue

        # Layer 2
        num_cross_overs = randint(1, len(gene1.w_rec_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_rec_readgate) - 1)
                gene1.w_rec_readgate[ind_cr, :] = gene2.w_rec_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_rec_readgate) - 1)
                gene2.w_rec_readgate[ind_cr, :] = gene1.w_rec_readgate[ind_cr, :]
            else:
                continue

        # Layer 3
        num_cross_overs = randint(1, len(gene1.w_mem_readgate))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_mem_readgate) - 1)
                gene1.w_mem_readgate[ind_cr, :] = gene2.w_mem_readgate[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_mem_readgate) - 1)
                gene2.w_mem_readgate[ind_cr, :] = gene1.w_mem_readgate[ind_cr, :]
            else:
                continue

        # OUTPUT WEIGHTS
        num_cross_overs = randint(1, len(gene1.w_output))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, len(gene1.w_output) - 1)
                gene1.w_output[ind_cr, :] = gene2.w_output[ind_cr, :]
            elif rand < 0.66:
                ind_cr = randint(0, len(gene1.w_output) - 1)
                gene2.w_output[ind_cr, :] = gene1.w_output[ind_cr, :]
            else:
                continue

        # MEMORY CELL (PRIOR)
        # 1-dimensional so point crossovers
        num_cross_overs = randint(1, int(gene1.w_rec_readgate.shape[1] / 2))
        for i in range(num_cross_overs):
            rand = random.random()
            if rand < 0.33:
                ind_cr = randint(0, gene1.w_rec_readgate.shape[1] - 1)
                gene1.w_rec_readgate[0, ind_cr:] = gene2.w_rec_readgate[0, ind_cr:]
            elif rand < 0.66:
                ind_cr = randint(0, gene1.w_rec_readgate.shape[1] - 1)
                gene2.w_rec_readgate[0, :ind_cr] = gene1.w_rec_readgate[0, :ind_cr]
            else:
                continue

        if self.num_substructures == 13:  # Only for NTM
            # WRITE GATES
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_writegate) - 1)
                    gene1.w_writegate[ind_cr, :] = gene2.w_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_writegate) - 1)
                    gene2.w_writegate[ind_cr, :] = gene1.w_writegate[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_rec_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                    gene1.w_rec_writegate[ind_cr, :] = gene2.w_rec_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                    gene2.w_rec_writegate[ind_cr, :] = gene1.w_rec_writegate[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_mem_writegate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                    gene1.w_mem_writegate[ind_cr, :] = gene2.w_mem_writegate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                    gene2.w_mem_writegate[ind_cr, :] = gene1.w_mem_writegate[ind_cr, :]
                else:
                    continue

    def regularize_weight(self, weight):
        if weight > self.parameters.weight_magnitude_limit:
            weight = self.parameters.weight_magnitude_limit
        if weight < -self.parameters.weight_magnitude_limit:
            weight = -self.parameters.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.0

        #Macro propoerties of weights










        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        for ssne_prob, key in zip(ss_mut_dist, keys): #For each structure
            if random.random()<ssne_prob:

                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                      W[key][
                                                                                          ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                                                                                          ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                        W[key][ind_dim1, ind_dim2])

    def epoch(self, pop, fitnesses):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitnesses);
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.parameters.pop_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning eleitst candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            pop[i] = copy.deepcopy(pop[off_i])
            pop[j] = copy.deepcopy(pop[off_j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.parameters.pop_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.mutation_prob:
                    self.mutate_inplace(pop[i])


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

