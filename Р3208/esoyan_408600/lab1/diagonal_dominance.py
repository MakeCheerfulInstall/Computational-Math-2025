from copy import deepcopy


def make_matrix_diagonal_dominance(matrix, values):
    n = len(matrix)
    mt = [-1 for _ in range(n)]
    used = []
    graph = []

    def try_kuhn(v):
        if used[v]:
            return False
        used[v] = True

        for h in graph[v]:
            if mt[h] == -1 or try_kuhn(mt[h]):
                mt[h] = v
                return True

        return False

    for i in range(n):
        row_sum_of_abs = sum(map(abs, matrix[i]))
        indexes = [j for j, x in enumerate(matrix[i]) if 2 * abs(x) >= row_sum_of_abs]
        graph.append(indexes)

    for v in range(n):
        used = [False for _ in range(n)]
        try_kuhn(v)

    new_matrix = deepcopy(matrix)
    new_values = deepcopy(values)

    for i in range(n):
        if mt[i] == -1:
            return False
        matrix[i] = new_matrix[mt[i]]
        values[i] = new_values[mt[i]]

    return True
