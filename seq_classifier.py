import numpy as np, os, math, random
import mod_ecj as mod, sys
from random import randint

class Tracker(): #Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 10: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
        self.pop_size = 100
        self.load_colony = 0
        self.total_gens = 20000

        #NN specifics
        self.num_hnodes = 5
        self.num_mem = 5
        self.grumb_topology = 1 #1: Default (hidden nodes cardinality attached to that of mem (No trascriber))
                                #2: Detached (Memory independent from hidden nodes (transcribing function))
                                #3: FF (Normal Feed-Forward Net)
        #self.output_activation = 'tanh' #tanh or hardmax

        #SSNE stuff
        self.elite_fraction = 0.1
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.extinction_prob = 0.00 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 1000000000000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        #Task Params
        self.depth = 7
        self.noise_len = [10,20]
        self.train_evals= 10
        self.valid_evals = 50

        #Dependents
        self.num_input = 1; self.num_output = 1
        if self.grumb_topology == 1: self.num_mem = self.num_hnodes
        self.save_foldername = 'R_ECJ/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class Sequence_classifier:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne = mod.SSNE(parameters)
        self.depth = parameters.depth; self.noise_len = self.parameters.noise_len
        self.pop = []
        for _ in range(parameters.pop_size):
            self.pop.append(mod.GRUMB(parameters))


    def generate_task(self, num_instances):
        input_set = []
        for _ in range(num_instances):
            input = []
            for i in range(self.depth):
                #Encode the signal (1 or -1s)
                if random.random() < 0.5: input.append(-1)
                else: input.append(1)
                if i == self.depth - 1: continue

                #Encdoe the noise (0's)
                num_noise = randint(self.noise_len[0], self.noise_len[1])
                for i in range(num_noise): input.append(0)
            input_set.append(input)

        return input_set

    def get_reward(self, input, output, is_test):
        if is_test: #Testing criteria (harsh criterion)
            target = 0.0
            reward = 1.0
            for i, j in zip(input, output):
                target += i
                if i == 1 or i == -1:
                    point_reward = j * target
                    if point_reward < 0:
                        reward = 0.0
                        break

        else: #Training criteria
            reward = 0.0
            target = 0.0
            for i, j in zip(input, output):
                target += i
                point_reward = j * target
                if point_reward > 1: point_reward = 1
                elif point_reward < -1: point_reward = -1
                reward += point_reward

        return reward

    def run_simulations(self, net, epoch_inputs, is_test):
        reward = 0.0
        for input in epoch_inputs:
            net.reset()
            net_output = []
            for inp in input: #Run network to get output
                inp = np.array([inp])
                net_output.append((2*(net.feedforward(inp)[0][0]-0.5)))
            reward += self.get_reward(input, net_output, is_test) #get reward or fitness of the individual

        return reward/len(epoch_inputs)

    def evolve(self):
        fitnesses = []

        #Generate train sets for the epoch
        train_set = self.generate_task(self.parameters.train_evals)

        for net in self.pop: #Test all genomes/individuals
            fitness = self.run_simulations(net, train_set, is_test=False)
            fitnesses.append(fitness)

        #Validation Score
        valid_set = self.generate_task(self.parameters.valid_evals)
        champion_index = fitnesses.index(max(fitnesses))
        valid_fitness = self.run_simulations(self.pop[champion_index], valid_set, is_test=True)

        self.ssne.epoch(self.pop, fitnesses)

        return fitnesses[champion_index], valid_fitness



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    if parameters.load_colony:
        gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
        tracker = mod.unpickle(parameters.save_foldername + 'tracker')
    else:
        tracker = Tracker(parameters, ['best_train', 'valid'], 'ecj.csv')  # Initiate tracker
        gen_start = 1
    print 'ECJ Training with', parameters.num_hnodes, 'hidden_nodes', parameters.num_mem, 'memory'
    sim_task = Sequence_classifier(parameters)
    for gen in range(gen_start, parameters.total_gens):
        best_train_fitness, validation_fitness = sim_task.evolve()
        print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.all_tracker[1][1]
        tracker.update([best_train_fitness, validation_fitness], gen)















