import pygame, random, copy
import numpy as np
import tetris as base
import AI as ai

seed = 50
random.seed(seed)
np.random.seed(seed)

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.score = 0

    def fitness_cal(self, state_of_game):
        self.score = state_of_game[2]

    def best_move_cal(self, board, piece, game=False):
        X, R, Y = 0, 0, 0
        best_score = -100000
        num_holes, num_blocks = base.calc_initial_move_info(board)
        for r in range(len(base.PIECES[piece['shape']])):
            for x in range(-2, base.BOARDWIDTH - 2):
                # [True(valid), max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
                info = base.calc_move_info(board, piece, x, r, num_holes, num_blocks)
                if info[0]:
                    score = 0
                    for i in range(1, len(info)):
                        score += self.genes[i - 1] * info[i]
                    if score > best_score:
                        best_score = score
                        X = x
                        R = r
                        Y = piece['y']
        if game:
            piece['y'] = Y
        else:
            piece['y'] = -2

        piece['x'] = X
        piece['rotation'] = R

        return X, R


class GeneticAlgorithm:
    def __init__(self, pop_size, genes_num=7):
        self.chromosomes = []
        for i in range(pop_size):
            genes = np.random.uniform(-1, 1, size=genes_num)
            chromosome = Chromosome(genes)
            self.chromosomes.append(chromosome)
            game = ai.run_game(self.chromosomes[i], no_show=True)
            self.chromosomes[i].fitness_cal(game)

    def information(self):
        generation = []
        for i, chromo in enumerate(self.chromosomes):
            generation.append((chromo.genes, chromo.score))
        return generation

    def selection(self):
        fits = np.array([chromosome.score for chromosome in self.chromosomes])
        normalized_f = fits / (fits.sum() + 1)
        cumulative_prob = np.cumsum(normalized_f)
        selected_pop = []

        while len(selected_pop) < len(self.chromosomes):
            R = random.random()
            for i, cumulative_value in enumerate(cumulative_prob):
                if R < cumulative_value:
                    selected_pop.append(copy.deepcopy(self.chromosomes[i]))
                    break
        return selected_pop

    def crossover(self, selected_pop, pc=0.7):
        new_pop = []
        pop_size = len(selected_pop)
        parents = []
        for k in range(pop_size):
            R = random.random()
            if R < pc:
                parents.append(selected_pop[k])
        if not parents:
            return copy.deepcopy(selected_pop)

        num_parents = len(parents)

        for i in range(0, num_parents - 1, 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            crossover_point = random.randint(1, len(p1.genes) - 1)
            c1_genes = np.concatenate([p1.genes[:crossover_point], p2.genes[crossover_point:]])
            c2_genes = np.concatenate([p2.genes[:crossover_point], p1.genes[crossover_point:]])
            new_pop.append(Chromosome(c1_genes))
            new_pop.append(Chromosome(c2_genes))

        if num_parents % 2 != 0:
            new_pop.append(copy.deepcopy(selected_pop[-1]))

        if len(new_pop) < pop_size:
            while len(new_pop) < pop_size:
                new_pop.append(copy.deepcopy(selected_pop[-1]))

        return new_pop

    def mutation(self, chromosomes, pm):
        num_genes = len(self.chromosomes[0].genes)
        pop_size = len(self.chromosomes)
        total_genes = num_genes * pop_size
        num_mutations = int(pm * total_genes)
        for _ in range(num_mutations):
            random_pos = random.randint(0, total_genes - 1)
            chr_index = random_pos // num_genes
            gene_index = random_pos % num_genes
            chromosomes[chr_index].genes[gene_index] = random.uniform(-1, 1)
        return chromosomes

    def replacement(self, new_chromosomes):
        combined_pop = self.chromosomes + new_chromosomes
        sorted_combined = sorted(combined_pop, key=lambda x: x.score, reverse=True)
        self.chromosomes = sorted_combined[:len(self.chromosomes)]

    def logFile(self, generation, gen_num, best_2_chromosomes):
        with open("logFile.txt", "a") as file:
            file.write("------------------------------\n")
            file.write(f"Generation {gen_num}\n")
            for i, (genes, score) in enumerate(generation):
                file.write(f"Chromosome {i + 1}\n")
                file.write(f"   Genes: {list(genes)}\n")
                file.write(f"   Score: {score}\n")
            file.write("the best 2 chromosomes in this generation: \n")
            for i, chromo in enumerate(best_2_chromosomes):
                file.write(f"Best Chromosome {i + 1}\n")
                file.write(f"   Genes: {list(chromo.genes)}\n")
                file.write(f"   Score: {chromo.score}\n")
            file.write("------------------------------\n")
            file.write("\n")

    def bestOverallLogFile(self, best_overall):
        file = open("logFile.txt", "a")
        file.write(f"Best Chromosome Overall Genes: {list(best_overall.genes)}\n")
        file.write(f"Best Chromosome Overall Score: {best_overall.score} \n")
    
    