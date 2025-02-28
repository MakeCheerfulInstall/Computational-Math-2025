import random

for i in range(20):
    for j in range(21):
        if i == j:
            print(round(random.random() * 10000, 3), end=' ')
        else:
            print(round((random.random() - 0.5) * 100, 3), end=' ')
    print()