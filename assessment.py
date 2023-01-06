import hw5
import matplotlib.pyplot as plt

file_names = ['hypothyroid.csv','mnist_1000.csv','monks1.csv','votes.csv']

def make_plot(random_seeds,dataset_list,plot_title):
    # print("random seeds:",random_seeds)
    # print("Neural Network accuracies:",dataset_list[0])
    plt.plot(random_seeds,dataset_list[0],label="Neural Network Accuracy",linestyle="--")
    plt.plot(random_seeds,dataset_list[1],label="Decision Tree Accuracy",linestyle="--")
    plt.plot(random_seeds,dataset_list[2],label="Naive Bayes Accuracy",linestyle="--")

    plt.legend()
    plt.show

def main():
    random_seeds = []

    hypo_accuracies = [[],[],[]]
    mnist_accuracies = [[],[],[]]
    monks_accuracies = [[],[],[]]
    votes_accuracies = [[],[],[]]

    for i in range(2):
        print("====ITERATION #",i+1,"====")
        return_list = hw5.run(i)

        random_seeds.append(i)

        #ORDER:
        #return list should be a list (each dataset) 
        # of lists (each model) 
        # of lists (accuracies for that model)

        for i in range(len(return_list[0])):
            hypo_accuracies[i].append(return_list[0][i][0])
        print("hypo accuracies",hypo_accuracies)

        for i in range(len(return_list[1])):
            mnist_accuracies[i].append(return_list[1][i][0])
        print("mnist accuracies:",mnist_accuracies)

        for i in range(len(return_list[2])):
            monks_accuracies[i].append(return_list[2][i][0])
        print("monks accuracies:",monks_accuracies)

        for i in range(len(return_list[3])):
            votes_accuracies[i].append(return_list[3][i][0])
        print("votes accuracies:",votes_accuracies)

    print("displaying plot for Hypothyroid")
    make_plot(random_seeds,hypo_accuracies,"Hypothyroid Data Set")

    print("displaying plot for mnist")
    make_plot(random_seeds,mnist_accuracies,"mnist_100 Data Set")

    print("displaying plot for monks")
    make_plot(random_seeds,monks_accuracies,"monks1 Data Sset")

    print("displaying plot for votes")
    make_plot(random_seeds,votes_accuracies,"votes Data Set")

main()