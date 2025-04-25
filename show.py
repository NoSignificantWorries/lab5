import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data")
parser.add_argument("-i", "--image")

args = parser.parse_args()
data_path = args.data
image_path = args.image

data = {}
with open(data_path, "r") as file:
    lines = file.readlines()[1:]
    lines = [line[:-1].split(" ") for line in lines]
    X = list(map(int, lines[0]))
    for line in lines[1:]:
        data[line[0]] = [[], []]
        for elem in line[1:]:
            elem = tuple(map(float, elem.split(",")))
            data[line[0]][0].append(elem[0])
            data[line[0]][1].append(elem[1])

plt.grid(True)
plt.xlabel("p")
plt.ylabel("Sp")

plt.plot(X, X, "g--", label="Linear")
for key in data.keys():
   plt.plot(X, data[key][1], label=key)

plt.legend()

plt.savefig(image_path, format="png", dpi=300)
