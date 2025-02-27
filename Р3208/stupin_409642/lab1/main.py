import copy
import numpy as np

# ---------------------------------------------GENERAL UTILS---------------------------------------------------------

def exit_with_error(message, file=None):
    print(message)
    if file != None:
        file.close()
    exit(1)

def concatenate_matrix_and_column(matrix, column):
    return [row + [column[i]] for i, row in enumerate(matrix)]

def split_matrix(matrix):
    a = [row[:-1] for row in matrix]
    b = [row[-1] for row in matrix]
    return a, b

def _round(n, precision):
    return "{:.{}f}".format(n, precision)

def to_lower_unicode(n):
    s = ""
    for c in str(n):
        s += chr(8320 + int(c))
    return s

# -----------------------------------------------IO UTILS------------------------------------------------------------

def read_n(file=None):
    try:
        if file == None: n = int(input("Введите n: "))
        else: n = int(file.readline())
    except:
        exit_with_error("n должно быть числом!", file)
    if n < 1 or n > 20:
        exit_with_error("Размер матрицы должен быть от 1 до 20 включительно!", file)
    return n

def read_matrix(n, file=None):
    matrix = []
    if file == None: print("Введите матрицу построчно (и A и B):")
    for _ in range(n):
        try:
            if file == None: row = list(map(float, input().split()))
            else: row = list(map(float, file.readline().split()))
        except:
            exit_with_error("Элементы матрицы должны быть числами!", file)
        if len(row) != n + 1:
            exit_with_error("В каждой строке матрицы должно быть n+1 элементов!", file)
        matrix.append(row)
    if len(matrix) != n:
        exit_with_error("В матрице должно быть n строк!", file)
    return matrix

def read_mode():
    print("Выберите способ ввода:")
    print("1 - ввод в консоли")
    print("2 - ввод из файла")
    print("3 - генерация матрицы")
    try:
        mode = int(input())
    except:
        exit_with_error("Режим должен быть числом!")
    if mode < 1 or mode > 3:
        exit_with_error("Режим должен быть от 1 до 3 включительно!")
    return mode

def print_matrix(matrix):
    for r in matrix:
        for c in r:
            print(_round(c, 2), end='\t')
        print()

def print_vector(vector, sym):
    for i in range(len(vector)):
        print(f'{sym}{to_lower_unicode(i + 1)} = {_round(vector[i], 2)}')
    print()

def print_system(a, b):
    n = len(a)
    for i in range(n):
        equation = " + ".join(f"{_round(a[i][j], 2)}x{to_lower_unicode(j + 1)}" for j in range(n))
        equation += f" = {_round(b[i], 2)}"
        print(equation)

# ----------------------------------------MATRIX UTILS-----------------------------------------------------------

def transform_matrix_to_triangle(matrix):
    n = len(matrix) 
    m = len(matrix[0])
    cnt = 0
    for i in range(n):
        if matrix[i][i] == 0:
            l = i
            for k in range(i+1, n):
                if matrix[k][i] != 0:
                    l = k
                    break
            if l != i:
                cnt += 1
                for j in range(i, m):
                    matrix[i][j], matrix[l][j] = matrix[l][j], matrix[i][j]
            else:
                continue
        for k in range(i + 1, n):
            c = matrix[k][i] / matrix[i][i]
            matrix[k][i] = 0
            for j in range(i + 1, m):
                matrix[k][j] = matrix[k][j] - matrix[i][j] * c
    return cnt

def det(matrix):
    n = len(matrix)
    if n != len(matrix[0]):
        exit_with_error("Матрица должна быть квадратной!")
    tmp = copy.deepcopy(matrix)
    k = transform_matrix_to_triangle(tmp)
    det = 1
    for i in range(n):
        det *= tmp[i][i]
    return det * (-1)**k

def rang(matrix):
    tmp = copy.deepcopy(matrix)
    transform_matrix_to_triangle(tmp)
    return sum(any(c != 0 for c in r) for r in tmp)

def calculate_r(a, b, x):
    n = len(a)
    r = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(n):
            s = s + a[i][j] * x[j]
        r[i] = s - b[i]
    return r

# -----------------------------------------------GAUSS METHOD-------------------------------------------------------

def gauss_direct_route(a, b):
    n = len(a)
    for i in range(n - 1):
        if a[i][i] == 0:
            l = i
            for m in range(i+1, n):
                if a[m][i] != 0:
                    l = m
                    break
            if l != i:
                for j in range(i, n):
                    a[i][j], a[l][j] = a[l][j], a[i][j]
                b[i], b[l] = b[l], b[i]
            else:
                continue
        for k in range(i + 1, n):
            c = a[k][i] / a[i][i]
            a[k][i] = 0
            for j in range(i + 1, n):
                a[k][j] = a[k][j] - a[i][j] * c
            b[k] = b[k] - b[i] * c

def gauss_reverse_route(a, b):
    n = len(a)
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):   
            s = s + a[i][j] * x[j]
        if a[i][i] == 0:
            if b[i] != 0:
                exit_with_error("СЛАУ не имеет решений!")
            else:
                exit_with_error("СЛАУ имеет бесконечно много решений!")
        x[i] = (b[i] - s) / a[i][i]
    return x

# -----------------------------------------------EQUATION SOLVERS-------------------------------------------------------

def gauss_equation_solver(a, b):
    n = len(a)
    ext_matrx = concatenate_matrix_and_column(a, b)

    matrix_rang = rang(a)
    ext_matrx_rang = rang(ext_matrx)

    if matrix_rang != ext_matrx_rang:
        exit_with_error("СЛАУ не имеет решений!")
    if matrix_rang < n:
        exit_with_error("СЛАУ имеет бесконечно много решений!")

    gauss_direct_route(a, b)

    triangle_matrix = concatenate_matrix_and_column(a, b)

    x = gauss_reverse_route(a, b)
    return x, triangle_matrix

def numpy_equation_solver(a, b):
    a = np.array(a)
    b = np.array(b)
    x = np.linalg.solve(a, b)
    return x

def generate_random_matrix(n):
    a = []
    b = []
    for i in range(n):
        c = []
        for j in range(n):
            c.append(np.random.rand() * 10)
        a.append(c)
        b.append(np.random.rand() * 10)
    return a, b

# ----------------------------------------------------MAIN-----------------------------------------------------------

def main():
    print("Решение СЛАУ методом Гаусса")
    mode = read_mode()
    file = None
    if mode == 1:
        print("Выбран ввод в консоли")
    elif mode == 2:
        print("Выбран ввод из файла")
        filename = input("Введите имя файла: ")
        try:
            file = open(filename)
        except:
            exit_with_error("Файл не найден!")
    elif mode == 3: 
        print("Выбран режим генерации случайной матрицы")
    
    n = read_n(file=file)
    if mode != 3:
        matrix = read_matrix(n, file=file)
        a, b = split_matrix(matrix)
    else:
        a, b = generate_random_matrix(n)

    if mode == 2:
        file.close()

    print("Считанная система:")
    print_system(a, b)

    my_det = det(a)
    print(f"Вычисленный определитель матрицы: {_round(my_det, 2)}")

    numpy_det = np.linalg.det(a)
    print(f"Определитель матрицы через numpy: {_round(numpy_det, 2)}")

    x_gauss, triangle_matrix = gauss_equation_solver(a, b)
    print("Треугольная матрица:")
    print_matrix(triangle_matrix)
    print("Решение системы методом Гаусса:")
    print_vector(x_gauss, 'x')

    x_numpy = numpy_equation_solver(a, b)
    print("Решение системы через numpy:")
    print_vector(x_numpy, 'x')

    r = calculate_r(a, b, x_gauss)
    print("Невязка:")
    print_vector(r, 'r')

if __name__ == "__main__":
    main()
