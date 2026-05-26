import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


episodes = []
scores = []
average_100 = []

with open("runs/rl/dqn_scores.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        episodes.append(int(row["episode"]))
        scores.append(float(row["score"]))
        average_100.append(float(row["average_100"]))


plt.figure()
plt.plot(episodes, scores, label="Episode score")
plt.plot(episodes, average_100, label="100-episode average")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN LunarLander Training Performance")
plt.legend()
plt.grid(True)

os.makedirs("runs/summary", exist_ok=True)
plt.savefig("runs/summary/dqn_training_curve.png")
plt.close()