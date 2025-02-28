def matrix_from_file():
    while True:
        try:
            file_name = input("Введите название файла: ")
            with open(file_name, 'r') as file:
                lines = list(map(lambda x: x.rstrip('\n'), file.readlines()))
                matrix = []
                for line in lines:
                    matrix.append(list(map(float, line.split())))
                return matrix
        except Exception:
            print("Файл некорректен")


def matrix_from_console():
    while True:
        try:
            n = int(input("Введите размер матрицы: "))
            if n < 1 or n > 20:
                print("Ввод некорректен!")
            else:
                break
        except ValueError:
            print("Ввод некорректен!")
    matrix = []
    print("Введите строки матрицы: ")
    for i in range(n):
        while True:
            try:
                matrix.append([float(x) for x in input().split()])
                break
            except ValueError:
                print("Введена некорректная строка!")
    return matrix


def get_matrix():
    source = ""
    while source != "1" and source != "2":
        source = input("Откуда будет вводиться матрица? 1 - консоль, 2 - файл ")
    if source == "2":
        return matrix_from_file()
    return matrix_from_console()


def get_accuracy():
    while True:
        try:
            a = float(input("Введите точность: "))
            if a > 0:
                return a
            else:
                print("Ввод некорректен!")
        except Exception:
            print("Ввод некорректен!")


def print_matrix(matrix):
    for row in matrix:
        print(*(f"{num:10.2f}" for num in row), sep=' ')


def print_x_values(x_values, precision):
    print(*list(map(lambda x: round(x, precision), x_values)))


def print_errors(x_last, x_current, precision):
    errors = [round(abs(x_last[i] - x_current[i]), precision) for i in range(len(x_last))]
    print(*errors)