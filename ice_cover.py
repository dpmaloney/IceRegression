import csv
import math
import numpy as np
def get_dataset():
    with open("Dataset.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        dataset = []
        for row in reader:
            if row[1] != "---":
                dataset.append([int(row[0].split("-")[0]), int(row[1])])

        return dataset

def print_stats(dataset):
    print(len(dataset))
    print(round(np.mean(dataset, axis=0)[1], 2))
    print(round(np.std(dataset, axis=0)[1], 2))


def regression(beta_0, beta_1):
    dataset = get_dataset()
    number = 0
    for point in dataset:
        number += math.pow((beta_0 + beta_1*point[0]-point[1]), 2)

    return number/len(dataset)

def regression_normalized(beta_0, beta_1):
    dataset = get_dataset()
    xmean = np.mean(dataset, axis=0)[0]
    xstd = np.std(dataset, axis=0)[0]
    for point in dataset:
        point[0] = (point[0] - xmean) / xstd

    number = 0
    for point in dataset:
        number += math.pow((beta_0 + beta_1*point[0]-point[1]), 2)

    return number/len(dataset)

def gradient_descent(beta_0, beta_1):
    dataset = get_dataset()
    n1 = 0
    n2 = 0
    for point in dataset:
        n1 += beta_0 + beta_1*point[0]-point[1]
        n2 += (beta_0+beta_1*point[0]-point[1])*point[0]

    return (n1*2/len(dataset), n2*2/len(dataset))


def iterate_gradient(T, eta):
    dataset = get_dataset()
    b0 = 0
    b1 = 0
    for t in range(T):
        n1 = 0
        n2 = 0
        for point in dataset:
            n1 += b0 + b1 * point[0] - point[1]
            n2 += (b0 + b1 * point[0] - point[1]) * point[0]

        mseb0 = (2/len(dataset))*n1
        mseb1 = (2/len(dataset))*n2


        b0 = b0 - eta*mseb0
        b1 = b1 - eta*mseb1
        print(t+1, round(b0, 2), round(b1, 2), round(regression(b0,b1), 2))


def compute_betas():
    dataset = get_dataset()
    means = np.mean(dataset, axis=0)
    xmean = means[0]
    ymean = means[1]
    b1num = 0
    b1denom = 0
    for point in dataset:
        if point[0]-xmean != 0:
            b1num += (point[0]-xmean)*(point[1]-ymean)
            b1denom += math.pow((point[0]-xmean), 2)

    b1 = b1num/b1denom

    b0 = ymean-b1*xmean

    return b0, b1, regression(b0, b1)


def predict(year):
    b0, b1, mse = compute_betas()

    return b0 + b1*year

def iterate_normalized(T, eta):
    dataset = get_dataset()
    xmean = np.mean(dataset, axis=0)[0]
    xstd = np.std(dataset, axis=0)[0]
    for point in dataset:
        point[0] = (point[0]-xmean)/xstd

    b0 = 0
    b1 = 0
    for t in range(T):
        n1 = 0
        n2 = 0
        for point in dataset:
            n1 += b0 + b1 * point[0] - point[1]
            n2 += (b0 + b1 * point[0] - point[1]) * point[0]

        mseb0 = (2 / len(dataset)) * n1
        mseb1 = (2 / len(dataset)) * n2

        b0 = b0 - eta * mseb0
        b1 = b1 - eta * mseb1
        print(t + 1, round(b0, 2), round(b1, 2), round(regression_normalized(b0, b1), 2))


def sgd(T, eta):
    dataset = get_dataset()
    xmean = np.mean(dataset, axis=0)[0]
    xstd = np.std(dataset, axis=0)[0]
    for point in dataset:
        point[0] = (point[0] - xmean) / xstd

    b0 = 0
    b1 = 0
    for t in range(T):
        n1 = 0
        n2 = 0
        import random
        point = random.choice(dataset)
        n1 += b0 + b1 * point[0] - point[1]
        n2 += (b0 + b1 * point[0] - point[1]) * point[0]

        mseb0 = (2 / len(dataset)) * n1
        mseb1 = (2 / len(dataset)) * n2

        b0 = b0 - eta * mseb0
        b1 = b1 - eta * mseb1
        print(t + 1, round(b0, 2), round(b1, 2), round(regression_normalized(b0, b1), 2))

sgd(5000, .1)