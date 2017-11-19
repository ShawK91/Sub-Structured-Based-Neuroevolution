import numpy as np, os
#import MultiNEAT as NEAT
import mod_mem_net as mod, sys
from random import randint
import random
#np.seterr(all='raise')
save_foldername = 'RSeq_classifier'
class tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        if parameters.is_memoried:
            self.file_save = 'mem_seq_classifier.csv'
        else:
            self.file_save = 'norm_seq_classifier.csv'


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/rough_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/hof_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self, is_memoried):
        self.num_input = 1
        self.num_hnodes = 5
        self.num_output = 1
        if is_memoried: self.type_id = 'memoried'
        else: self.type_id = 'normal'

        self.elite_fraction = 0.1
        self.crossover_prob = 0
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            #Normalize network flexibility by changing hidden nodes
            naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
            mem_weights = 3 * (
                 self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                 self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
            normalization_factor = int(mem_weights/naive_total_num_weights)

            #Set parameters for comparable flexibility with memoried net
            self.num_hnodes *= normalization_factor + 1
            self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.depth = 21
            self.interleaving_lower_bound = 10
            self.interleaving_upper_bound = 20
            self.is_memoried = 1
            self.repeat_trials = 10
            self.test_trials = 50

            #DEAP/SSNE stuff
            self.use_ssne = 1
            self.use_deap = 0
            if self.use_deap or self.use_ssne:
                self.ssne_param = SSNE_param( self.is_memoried)
            self.total_gens = 10000
            self.arch_type = 1 #0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture


            #Reward scheme
            #1 Block continous reward - End decision matters
            #2 Block reward binary - End decision matters plus also calculated binary rather than continously
            #3 Fine continous reward - prediction at each time-step matters
            #4 Coarse reward calcluated only at points of 1/-1 introdcution
            #5 Combine #3 and #2 (test)
            #6 Add #3 and #4
            self.reward_scheme = 3
            self.tolerance = 1
            self.test_tolerance = 1
            if self.arch_type == 0: self.arch_type = 'quasi_gru'
            elif self.arch_type ==1: self.arch_type = 'quasi_ntm'
            else: sys.exit('Invalid choice of neural architecture')

parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker

class Sequence_classifier:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.depth = self.parameters.depth
        self.interleaving_upper_bound = self.parameters.interleaving_upper_bound; self.interleaving_lower_bound = self.parameters.interleaving_lower_bound

        self.agent = mod.SSNE(self.parameters, self.ssne_param, parameters.arch_type)

    def generate_input(self):
        input = []
        for i in range(self.depth):
            #Encode the signal (1 or -1s)
            if random.random() < 0.5: input.append(-1)
            else: input.append(1)
            if i == self.depth - 1: continue

            #Encdoe the noise (0's)
            num_noise = randint(self.interleaving_lower_bound, self.interleaving_upper_bound)
            for i in range(num_noise): input.append(0)
        return input

    def get_reward(self, input, output):

        if self.parameters.reward_scheme == 1: #Block continous reward - End decision matters
            target = sum(input)
            if target > 1: target = 1
            elif target < -1: target = -1
            reward = output[-1] * target * 1.0

        elif self.parameters.reward_scheme == 2: #2 Block reward binary - End decision matters plus also calculated binary rather than continously
            target = sum(input)
            if target > 1: target = 1
            elif target < -1: target = -1
            if target * output[-1] > (1.0 - self.parameters.tolerance): reward = 1.0
            else: reward = 0.0


        elif self.parameters.reward_scheme == 3: #3 Fine continous reward - prediction at each time-step matters
            reward = 0.0
            target = 0.0
            for i, j in zip(input, output):
                target += i
                point_reward = j * target
                if point_reward > 1: point_reward = 1
                elif point_reward < -1: point_reward = -1
                reward += point_reward

        elif self.parameters.reward_scheme == 4: #4 Coarse reward clacluated only at points of 1/-1 introdcution
            reward = 0.0
            target = 0.0
            for i, j in zip(input, output):
                target += i
                if i == 1 or i == -1:
                    point_reward = j * target
                    if point_reward > 1: point_reward = 1
                    elif point_reward < -1: point_reward = -1
                    reward += point_reward

        elif self.parameters.reward_scheme == 5: #Combine #3 and test (#2)
            reward = 0.0
            target = 0.0
            for i, j in zip(input, output):
                target += i
                point_reward = j * target
                if point_reward > 1: point_reward = 1
                elif point_reward < -1: point_reward = -1
                reward += point_reward
            if j * target > 0: reward += parameters.depth / 2.0

        elif self.parameters.reward_scheme == 6: #3 Add #3 and #4
            reward = 0.0
            target = 0.0
            for i, j in zip(input, output):
                target += i
                point_reward = j * target
                if point_reward > 1: point_reward = 1
                elif point_reward < -1: point_reward = -1
                if i != 0 and point_reward == 1: reward += 1 #At =-1 points extra point
                reward += point_reward

        return reward

    def run_simulation(self, index, epoch_inputs):
        reward = 0.0
        for input in epoch_inputs:
            self.agent.pop[index].reset_bank()
            net_output = []
            for inp in input: #Run network to get output
                inp=np.array([inp])
                net_output.append((self.agent.pop[index].feedforward(inp)[0][0] - 0.5) * 2)
            reward += self.get_reward(input, net_output) #get reward or fitness of the individual

        #print net_output
        reward /= self.parameters.repeat_trials #Normalize
        self.agent.fitness_evals[index] = reward #Encode reward as fitness for individual
        return reward

    def evolve(self):
        best_epoch_reward = -1000000

        #Generate epoch input
        epoch_inputs = []
        for i in range(parameters.repeat_trials):
            epoch_inputs.append(self.generate_input())

        for i in range(self.parameters.population_size): #Test all genomes/individuals
            reward = self.run_simulation(i, epoch_inputs)
            if reward > best_epoch_reward: best_epoch_reward = reward

        #HOF test net
        hof_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        hof_score = self.test_net(hof_index)

        #Save population and HOF
        if (gen + 1) % 1000 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/seq_classification_pop')
            mod.pickle_object(self.agent.pop[hof_index], save_foldername + '/seq_classification_hof')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, hof_score

    def test_net(self, index):  # Test is binary
        reward = 0.0
        for trial in range(self.parameters.test_trials):
            self.agent.pop[index].reset_bank()
            if trial == self.parameters.test_trials - 1: print self.agent.pop[index].memory_cell.transpose(),

            input = self.generate_input()  # get input
            net_output = []
            for inp in input:  # Run network to get output
                inp = np.array([inp])
                net_output.append((self.agent.pop[index].feedforward(inp)[0][0] - 0.5) * 2)

            target = 0.0
            reward += 1

            for i, j in zip(input, net_output):
                target += i
                if i == 1 or i == -1:
                    point_reward = j * target
                    if point_reward < 0:
                        reward -= 1
                        break

            if trial == self.parameters.test_trials - 1: print target, net_output[-1]

        print self.agent.pop[index].memory_cell.transpose()
        print
        return reward / (self.parameters.test_trials)

if __name__ == "__main__":
    print 'Running SEQUENCE CLASSIFIER with ', parameters.arch_type
    task = Sequence_classifier(parameters)
    for gen in range(parameters.total_gens):
        epoch_reward, hof_score = task.evolve()
        print 'Generation:', gen+1, ' Epoch_reward:', "%0.2f" % epoch_reward, '  Score:', "%0.2f" % hof_score, '  Cumul_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(hof_score, gen)  # Add best global performance to tracker














