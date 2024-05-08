import AI as ai
import tetris as base
import GA as ga

def main():
    x = int(input("Enter 1 for training then test Or 2 for testing with best chromosome:"))
    base.main()
    if(x == 1):
        best_cs = ai.train_ai()
        print(best_cs.genes)
    else:
        best_cs = ga.Chromosome([-0.4039929986224746, -0.4753903548072371, -0.9897409018710861, 0.08640503056081195, 0.5141778421949406, 0.27274735514304527, 0.9049792038780815])
        res = ai.test_chromosome(best_cs)
        print(res)

if __name__ == "__main__":
    main()

# Best Chromosome Overall Genes: [-0.08283846596819111, -0.4206463319414011, -0.31248239545103385, -0.2601077187306704, 0.3101326967469451, -0.530528402492987, 0.14996717430201834]
# Best Chromosome Overall Score: 20275