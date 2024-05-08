import AI as ai
import tetris as base
import GA as ga

def main():
    x = int(input("Enter 1 for training then test Or 2 for testing with best chromosome:"))
    base.main()
    if(x == 1):
        best_cs = ai.train_ai()
    else:
        best_cs = ga.Chromosome([-0.1569083865955414, -0.9891202622783586, -0.9477969086575992, 0.045593164439093226, 0.5227476359230623, -0.22043626796392823, 0.3696077645687237])
        res = ai.test_chromosome(best_cs)
        print(res)

if __name__ == "__main__":
    main()