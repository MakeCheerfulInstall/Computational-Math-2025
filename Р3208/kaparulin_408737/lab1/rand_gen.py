import random

import numpy as np

n = int(input("Input n: "))
e = float(input("Input e: "))

with open("pars.txt", "w") as f:
    f.write(str(n) + "\n")
    f.write(str(e) + "\n")

    lines = []
    diag = list(range(n))

    for _ in range(n): 
        random_number = random.choice(diag)

        diag.remove(random_number)

        line = np.random.randint(0, 10, n+1)

        lines.append(" ".join(map(str, line)) + "\n")


    f.writelines(lines)

