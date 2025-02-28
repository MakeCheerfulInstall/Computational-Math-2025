import sys

from diagonal_dominance import make_matrix_diagonal_dominance
from user_interaction import get_matrix, get_accuracy, print_matrix, print_x_values, print_errors

LIMIT = 5000
PRECISION = 6


def converges(vars_old, vars_new):
    return sum((vars_new[i] - vars_old[i]) ** 2 for i in range(len(vars_old))) ** 0.5 < accuracy


def calculate_matrix_norm(matrix):
    max_row_sum = 0
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix[i])) if j != i) / matrix[i][i]
        max_row_sum = max(max_row_sum, row_sum)
    return max_row_sum


def iteration(coefficients, vars, values):
    n = len(coefficients)
    vars_new = [0.0] * n
    for i in range(n):
        s = sum(coefficients[i][j] * vars[j] for j in range(n) if i != j)
        vars_new[i] = (values[i] - s) / coefficients[i][i]
    return vars_new


def solve(coefs, values, precision=PRECISION):
    n = len(coefs)
    x = [0 for _ in range(n)]
    for i in range(1, LIMIT + 1):
        x_new = iteration(coefs, x, values)
        if converges(x, x_new):
            print('Решение:')
            print_x_values(x_new, precision)
            matrix_norm = calculate_matrix_norm(coefs)
            print(f"Норма матрицы (максимум сумм модулей строк): {matrix_norm}")
            print('Итераций:', i)
            print('Погрешности:')
            print_errors(x, x_new, precision)
            break
        x = x_new
    else:
        print("Видимо метод расходится...")


if __name__ == '__main__':
    accuracy = get_accuracy()
    matrix = get_matrix()
    coefs = []
    values = []
    for row in matrix:
        coefs.append(row[:-1])
        values.append(row[-1])
    if not make_matrix_diagonal_dominance(coefs, values):
        print("Матрица не может быть подготовлена для выполнения алгоритма")
        sys.exit()
    for i in range(len(matrix)):
        matrix[i] = coefs[i] + [values[i]]
    print("Преобразованная матрица:")
    print_matrix(matrix)
    solve(coefs, values)
