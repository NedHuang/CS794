# if __name__ == "__main__":
import csv
import matplotlib.pyplot as plt
def run_script():
    with open('iterations_gd.csv', mode='r') as iteration_file:
        # iters = iteration_file.readline()
        # mu_ = iteration_file.readline()
        iters = [float(i) for i in iteration_file.readline().split(',')]
        mu_ = [float(i) for i in iteration_file.readline().split(',')]
        print(iters)
        print(mu_)

    with open('iterations_accelerated_gd.csv', mode='r') as iteration_file:
        # iters = iteration_file.readline()
        # mu_ = iteration_file.readline()
        iters_ac = [float(i) for i in iteration_file.readline().split(',')]
        mu_ = [float(i) for i in iteration_file.readline().split(',')]
        print(iters)
        print(mu_)



    fig = plt.figure(figsize=(16, 12))
    plt.plot(mu_, iters, label=("Gradient Descent + Armijo"), linewidth=2.0, color ="black")
    plt.plot(mu_, iters_ac, label=("Accelerated + Armijo"), linewidth=2.0, color ="blue")
    plt.legend(prop={'size': 20},loc="upper right")
    plt.xlabel("mu", fontsize=25)
    plt.ylabel("num of iterations", fontsize=25)
    plt.grid(linestyle='dashed')
    # plt.show()
    plt.savefig('q7.png')

run_script()