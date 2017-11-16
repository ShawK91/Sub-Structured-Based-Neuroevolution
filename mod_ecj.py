from random import randint
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import LSTM, GRU, SimpleRNN
#from keras.layers.advanced_activations import SReLU
#from keras.regularizers import l2, activity_l2
#from keras.optimizers import SGD
#from deap import base
#from deap import creator
#from deap import tools
import math, copy
#import MultiNEAT as NEAT
import numpy as np, time
import random
from scipy.special import expit
#import pickle
#import neat as py_neat
#from neat import nn
import sys,os, cPickle
#TODO Extend distribution to crossover operation

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

#Add memory_write_gate
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

        #Forget gate
        self.w_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_forgetgate = np.mat(np.reshape(self.w_forgetgate, (num_hnodes, (num_input + 1))))
        self.w_rec_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_forgetgate = np.mat(np.reshape(self.w_rec_forgetgate, (num_hnodes, (num_output + 1))))
        self.w_mem_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_forgetgate = np.mat(np.reshape(self.w_mem_forgetgate, (num_hnodes, (num_hnodes + 1))))

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

        #Forget Gate
        ig_1 = self.linear_combination(self.w_forgetgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_forgetgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_forgetgate, last_memory)
        forget_gate_out = ig_1 + ig_2 + ig_3
        forget_gate_out = self.fast_sigmoid(forget_gate_out)

        #Memory Output
        memory_output = np.multiply(forget_gate_out, self.memory_cell)

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

    def reset_bank(self):
        self.last_output *= 0  # last output
        self.memory_cell = np.copy(self.bank_memory_cell) #Memory Cell

    def reset(self):
        self.last_output *= 0  # last output
        self.memory_cell = np.copy(self.bank_memory_cell) #Memory Cell



    def set_bank(self):
        #self.bank_last_output = self.last_output[:]  # last output
        self.bank_memory_cell = np.copy(self.memory_cell)  # Memory Cell

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

        #Forget Gates
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_forgetgate = weights[start:end]
        self.w_forgetgate = np.mat(np.reshape(w_forgetgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_forgetgate = weights[start:end]
        self.w_rec_forgetgate = np.mat(np.reshape(w_rec_forgetgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_forgetgate = weights[start:end]
        self.w_mem_forgetgate = np.mat(np.reshape(w_mem_forgetgate, (self.num_hnodes, (self.num_hnodes + 1))))

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

                # FORGET GATES
                # Layer 1
                num_cross_overs = randint(1, len(gene1.w_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                        gene1.w_forgetgate[ind_cr, :] = gene2.w_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                        gene2.w_forgetgate[ind_cr, :] = gene1.w_forgetgate[ind_cr, :]
                    else:
                        continue

                # Layer 2
                num_cross_overs = randint(1, len(gene1.w_rec_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                        gene1.w_rec_forgetgate[ind_cr, :] = gene2.w_rec_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                        gene2.w_rec_forgetgate[ind_cr, :] = gene1.w_rec_forgetgate[ind_cr, :]
                    else:
                        continue

                # Layer 3
                num_cross_overs = randint(1, len(gene1.w_mem_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                        gene1.w_mem_forgetgate[ind_cr, :] = gene2.w_mem_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                        gene2.w_mem_forgetgate[ind_cr, :] = gene1.w_mem_forgetgate[ind_cr, :]
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
                num_cross_overs = randint(1, int(gene1.w_rec_forgetgate.shape[1] / 2))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                        gene1.w_rec_forgetgate[0, ind_cr:] = gene2.w_rec_forgetgate[0, ind_cr:]
                    elif rand < 0.66:
                        ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                        gene2.w_rec_forgetgate[0, :ind_cr] = gene1.w_rec_forgetgate[0, :ind_cr]
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

            # Initiate distribution
            if self.parameters.mut_distribution == 1:  # Gaussian
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.parameters.mut_distribution == 2:  # Laplace
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.parameters.mut_distribution == 3:  # Uniform
                ss_mut_dist = np.random.uniform(0, 1, self.num_substructures)
            else:
                ss_mut_dist = [1] * self.num_substructures


            # INPUT GATES
            # Layer 1
            if random.random() < ss_mut_dist[0]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * gene.w_inpgate[
                            ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inpgate[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[1]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                               gene.w_rec_inpgate[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_rec_inpgate[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[2]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_mem_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_mem_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                               super_mut_strength *
                                                                               gene.w_mem_inpgate[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_mem_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_mem_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_mem_inpgate[ind_dim1, ind_dim2])

            # BLOCK INPUTS
            # Layer 1
            if random.random() < ss_mut_dist[3]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inp.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_inp.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_inp.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                       super_mut_strength * gene.w_inp[
                                                                           ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inp[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inp[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[4]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inp.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_inp.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_inp.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                           super_mut_strength * gene.w_rec_inp[
                                                                               ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inp[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_rec_inp[ind_dim1, ind_dim2])

            # FORGET GATES
            # Layer 1
            if random.random() < ss_mut_dist[5]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                              super_mut_strength *
                                                                              gene.w_forgetgate[
                                                                                  ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_forgetgate[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[6]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_rec_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                  gene.w_rec_forgetgate[
                                                                                      ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[7]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_mem_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_mem_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_mem_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                  gene.w_mem_forgetgate[
                                                                                      ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_mem_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2])


            # OUTPUT WEIGHTS
            if random.random() < ss_mut_dist[8]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_output.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_output.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_output.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_output[ind_dim1, ind_dim2] += random.gauss(0,
                                                                          super_mut_strength * gene.w_output[
                                                                              ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_output[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_output[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_output[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_output[ind_dim1, ind_dim2])

            # MEMORY CELL (PRIOR)
            if random.random() < ss_mut_dist[9]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = 0
                    ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                              super_mut_strength *
                                                                              gene.w_forgetgate[
                                                                                  ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_forgetgate[ind_dim1, ind_dim2])

            if random.random() < ss_mut_dist[10]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_writegate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_writegate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_writegate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                             super_mut_strength *
                                                                             gene.w_writegate[
                                                                                 ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                             mut_strength *
                                                                             gene.w_writegate[
                                                                                 ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_writegate[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[11]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_writegate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_writegate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_writegate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 super_mut_strength *
                                                                                 gene.w_rec_writegate[
                                                                                     ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                 gene.w_rec_writegate[
                                                                                     ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_rec_writegate[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[12]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_writegate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_mem_writegate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_mem_writegate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 super_mut_strength *
                                                                                 gene.w_mem_writegate[
                                                                                     ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                 gene.w_mem_writegate[
                                                                                     ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_mem_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_mem_writegate[ind_dim1, ind_dim2])

        def epoch(self, pop, fitnesses):
            # Reset the memory Bank the adaptive/plastic structures for all genomes
            for gene in pop:
                gene.reset_bank()

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

            # Bank the adaptive/plastic structures for all genomes with new changes
            for gene in pop:
                gene.set_bank()

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

