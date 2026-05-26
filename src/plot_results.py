import csv
import matplotlib.pyplot as plt



generations = []
best = []
average = []
overall_best = []
RUN_NAME = "default"

with open(f"runs/{RUN_NAME}/training_log.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        generations.append(int(row["generation"]))
        best.append(float(row["best"]))
        average.append(float(row["average"]))
        overall_best.append(float(row["overall_best"]))


plt.figure()
plt.plot(generations, best, label="Generation best")
plt.plot(generations, average, label="Generation average")
plt.plot(generations, overall_best, label="Overall best")

plt.xlabel("Generation")
plt.ylabel("Fitness / reward")
plt.title("Evolutionary Training Progress")
plt.legend()
plt.grid(True)

plt.savefig(f"runs/{RUN_NAME}/training_plot.png")
plt.show()