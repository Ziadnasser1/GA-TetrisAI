import time, pygame, copy
import tetris as base
import GA as ga
import AI as ai
import random
import numpy as np

seed = 50
random.seed(seed)
np.random.seed(seed)

GEN = 12
POP_SIZE = 15
PM = 0.2
PC = 0.6
ITERATIONS = 10000


def run_game(chromosome, speed=1000, iterations=ITERATIONS, max_score=50000, no_show=False):
    base.FPS = int(speed)
    base.main()

    board = base.get_blank_board()
    last_fall_time = time.time()
    score = 0
    level, fall_freq = base.calc_level_and_fall_freq(score)
    falling_piece = base.get_new_piece()
    next_piece = base.get_new_piece()

    chromosome.best_move_cal(board, falling_piece)

    num_used_pieces = 0
    removed_lines = [0, 0, 0, 0]

    alive = True
    win = False

    while alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        if falling_piece is None:
            falling_piece = next_piece
            next_piece = base.get_new_piece()

            chromosome.best_move_cal(board, falling_piece, no_show)

            num_used_pieces += 1
            score += 1

            last_fall_time = time.time()

            # if num_used_pieces >= iterations:
            #     alive = False

            if not base.is_valid_position(board, falling_piece):
                alive = False

        if no_show or time.time() - last_fall_time > fall_freq:
            if not base.is_valid_position(board, falling_piece, adj_Y=1):
                base.add_to_board(board, falling_piece)
                num_removed_lines = base.remove_complete_lines(board)
                # Bonus score for complete lines at once
                #                 # 40   pts for 1 line
                #                 # 120  pts for 2 lines
                #                 # 300  pts for 3 lines
                #                 # 1200 pts for 4 lines
                if num_removed_lines == 1:
                    score += 40
                    removed_lines[0] += 1
                    level+=1
                elif num_removed_lines == 2:
                    score += 120
                    removed_lines[1] += 1
                    level += 2
                elif num_removed_lines == 3:
                    score += 300
                    removed_lines[2] += 1
                    level += 3
                elif num_removed_lines == 4:
                    score += 1200
                    removed_lines[3] += 1
                    level += 4

                falling_piece = None
            else:
                falling_piece['y'] += 1
                last_fall_time = time.time()

        if not no_show:
            draw_game_on_screen(board, score, level, next_piece, falling_piece)

        if score > max_score:
            alive = False
            win = True

    game_state = [num_used_pieces, removed_lines, score, win]
    # print(num_used_pieces)

    return game_state


def draw_game_on_screen(board, score, level, next_piece, falling_piece):
    """Draw game on the screen"""

    base.DISPLAYSURF.fill(base.BGCOLOR)
    base.draw_board(board)
    base.draw_status(score, level)
    base.draw_next_piece(next_piece)

    if falling_piece != None:
        base.draw_piece(falling_piece)

    pygame.display.update()
    base.FPSCLOCK.tick(base.FPS)


def train_ai(generations_num=GEN, population_size=POP_SIZE , Pm=PM):
    pop = ga.GeneticAlgorithm(population_size)
    best_chromosomes = []

    for gen in range(generations_num):

        selected_population = pop.selection()

        new_pop = pop.crossover(selected_population)
        new_pop = pop.mutation(new_pop, Pm)

        for i in range(POP_SIZE):
            game_state = run_game(new_pop[i], iterations=ITERATIONS, no_show=True)
            new_pop[i].fitness_cal(game_state)

        pop.replacement(new_pop)

        sorted_chromosomes = sorted(pop.chromosomes, key=lambda c: c.score, reverse=True)

        best_2 = sorted_chromosomes[:2]
        best_chromosome = best_2[0]
        best_chromosomes.append(copy.deepcopy(best_chromosome))
        print(f"Generation {gen + 1}: Best Score: {best_chromosome.score}")
        gen_info = pop.information()
        pop.logFile(gen_info, gen + 1, best_2)

    best_overall_chromosome = max(best_chromosomes, key=lambda c: c.score)
    pop.bestOverallLogFile(best_overall_chromosome)

    return best_overall_chromosome


def test_chromosome(chromosome):
    print("Testing Phase started")
    game_state = ai.run_game(chromosome, iterations=ITERATIONS, no_show=False)
    return game_state
