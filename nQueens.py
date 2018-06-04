"""
p1-1: N Queens problem
algorithm of GA is Based on the paper named:
Nag, Sayan, and Uddalok Sarkar. "An Adaptive Genetic Algorithm for Solving N-Queens Problem." (2017).

python used to code and debug is python 3.6.3 using modules: numpy 1.14.2, matplotlib 2.2.2

"""
import csv
import time
import numpy as np
import matplotlib.pyplot as plt


start = time.clock()

x = 1
nQueens = 32	# number of queens (checkboard size)"

while nQueens < 4:
	try:
		nQueens = int(input('Please enter a number (more than 4): '))
	except ValueError:
		print('Dude, please Enter a number...')
		nQueens = 0

max_fit = 2 * (nQueens - 1)	# maximum fitness
pop_size = 1000	# population size
mut_chance = 0.8	# mutation rate
max_iter = 10000	# iteration limit for loop

def generateChromosome(nQueens):
	# init_dist is for making first chromosomes
	init_dist = np.arange(nQueens)
	np.random.shuffle(init_dist)
	return list(init_dist)

def generatePopulation(pop_size, nQueens):
	# initial population
	population = [generateChromosome(nQueens) for i in range(pop_size)]
	return population

# fitness function to evaluate each individual in population
def fitness(chromosome):
	f1 = list(map(lambda x: x[1] - x[0], enumerate(chromosome)))
	f2 = list(map(lambda x: 1 + len(chromosome) - x[1] - x[0], enumerate(chromosome)))
	# number of repetitive queens in one diagonal when seen from left corner(a,1)
	t1 = len(f1) - len(np.unique(f1))
	# number of repetitive queens in one diagonal when seen from right corner(h,1)
	t2 = len(f2) - len(np.unique(f2))
	return (max_fit - (t1 + t2)) / max_fit	# return normalized fitness value

# TODO: diversity evaluation

population = generatePopulation(pop_size, nQueens)
for i in range(len(population)): population[i].append(fitness(population[i]))

def mutation(child):
	for i in range(2):
		if np.random.random() < mut_chance:
			a, b = 0, 0
			while a == b: [a, b] = np.sort(np.round(np.random.rand(2)*(nQueens-1)))
			# swapping two arguments
			child[int(a)], child[int(b)] = child[int(b)], child[int(a)]
	return child

def crossover(*arg):
	twins = []
	a, b = 0, 0
	while a == b: [a, b] = np.sort(np.round(np.random.rand(2)*(nQueens-1)))
	chr1 = arg[0]
	chr2 = arg[1]
	for i in range(2):
		if i == 1:
			chr1 = arg[1]
			chr2 = arg[0]
		child = list(np.zeros(nQueens))
		child[int(a):int(b+1)] = chr1[int(a):int(b+1)]
		part = [x for x in chr2 if x not in chr1[int(a):int(b+1)]]
		child[:int(a)] = part[:int(a)]
		child[int(b+1):] = part[int(a):]
		# apply mutation on child
		mutation(child)
		# evaluate fitness of the child
		child.append(fitness(child))
		twins.append(child)
	return twins

def plot(*arg):
	min_fit = np.min(arg[0], axis = 0)[nQueens]
	max_fit = np.max(arg[0], axis = 0)[nQueens]
	mean_fit = np.mean(arg[0], axis = 0)[nQueens]
	elapse = time.clock() - start
	#printing values
	print('\nelapsed time: ', elapse)
	print('min fitness: ', min_fit)
	print('max fitness: ', max_fit)
	print('mean fitness: ', mean_fit)
	print('number of iteration: ', arg[-1])
	print('\nsolutions: ')
	for i in range(len(arg[1])):
		s = arg[1][i][:nQueens]
		print(s)

	#exporting to csv
	with open('./note1.csv', "a", newline='') as fp:
		wr = csv.writer(fp, dialect='excel')
		wr.writerow([min_fit, max_fit, mean_fit, elapse, arg[-1],
			    arg[1][0][:nQueens], arg[1][1][:nQueens], arg[1][2][:nQueens]])

	#plot for min_box, max_box,mean_box
	plt.figure(figsize=(13, 9), dpi=200)
	plt.plot(arg[-2], label='Min Fitness')
	plt.plot(arg[-3], label='Max Fitness')
	plt.plot(arg[-4], label='Mean Fitness', color='green')
	plt.xlabel('Iteration')
	plt.ylabel('Fitnesses value')
	plt.title('How fitness changes over iterations')
	plt.legend()
	plt.savefig('fig/solution{}-1.png'.format(x))
	plt.clf()

	#plot for new_gen_mean_box, mean_box
	plt.figure(figsize=(13, 9), dpi=200)
	plt.plot(arg[2], label='new generation mean Fitness')
	plt.plot(arg[-4], label='Mean Fitness', color='green')
	plt.xlabel('Iteration')
	plt.ylabel('Fitnesses value')
	plt.title('How fitness changes over iterations')
	plt.legend()
	plt.savefig('fig/solution{}-2.png'.format(x))
	plt.clf()

# main function
def selection(population):
	#count = 0
	new_gen_mean_box = []
	mean_box, max_box, min_box = [], [], []
	population.sort(key = lambda x: x[nQueens], reverse = True)	# sorted by best fitness
	for i in range(max_iter):
		new_generation = []
		# generating 1000 children
		for j in range(250):
			top_ten = population[:10]
			rest = population[10:]
			[a, b] = np.round(np.random.rand(2)*9)
			[c, d] = np.round(np.random.rand(2)*(len(population)-11))
			# choosing parents for crossover
			new_generation[4*j:4*j+2] = crossover(top_ten[int(a)][:nQueens], top_ten[int(b)][:nQueens])
			new_generation[4*j+2:4*j+4] = crossover(rest[int(c)][:nQueens], rest[int(d)][:nQueens])
		#adding new generations to population
		population.extend(new_generation)
		population.sort(key = lambda x: x[nQueens], reverse = True)	# sorted by best fitness
		# removing worst 4
		population = population[:pop_size]
		# saving each generations criteria
		new_gen_mean_box.append(np.mean([x[nQueens] for x in new_generation]))
		mean_box.append(np.mean(population, axis = 0)[nQueens])
		max_box.append(np.max(population, axis = 0)[nQueens])
		min_box.append(np.min(population, axis = 0)[nQueens])

		# adding the solutions to a list (if fitness is 1)
		solutions = list(filter(lambda x: x[nQueens] == 1, population))
		# making a list of unique solutions
		solutions = list(map(list, set(map(tuple, solutions))))
		#count += 1
		if len(solutions) > 3: break
	# plotting and other stuff
	if len(solutions) > 3: plot(population,solutions, new_gen_mean_box, mean_box, max_box, min_box, i)


def initial(t):
	selection(population)
	global x
	x = t

if __name__ == '__main__':
    initial(20)