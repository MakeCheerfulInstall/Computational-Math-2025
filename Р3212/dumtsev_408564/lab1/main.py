from random import random
import numpy as np
import traceback
import pandas as pd


class LinearEquationSystem:

    def __init__(self, matrix, vector, precision):
        self.matrix = matrix
        self.vector = vector
        self.precision = precision
        self.__solution = None
        self.initial_value = vector
        self.__precision_vector = []
        self.__iter_counter = 0
        self.__norm = 0
        if not self.ensure_diagonal():
            print("Matrix can't be transformed to neccessary condition.")
            raise SystemExit
        print("Matrix is: ", self.matrix)
        self.solve()
        self.calculate_norm()

    @property
    def norm(self):
        return self.__norm

    @property
    def solution(self):
        return self.__solution

    @property
    def iter_counter(self):
        return self.__iter_counter

    @property
    def precision_vector(self):
        return self.__precision_vector

    @staticmethod
    def generate_matrix(dim, max_num):
        matrix = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(random() * max_num)
            row.insert(i, sum(row[:-1]) + 1)

            matrix.append(row)
        return matrix

    def is_diag(self):
        # Checking, that for each row its diagonal element is greater than non-diagonal ones
        if all([abs(self.matrix.iloc[i, i]) >= sum([abs(elem) for idx, elem in enumerate(self.matrix.iloc[i]) if idx != i]) for i in range(len(self.vector))]):
            return True
        return False

    def ensure_diagonal(self):
        if self.is_diag():
            return True
        new_vector = np.zeros(len(self.vector))
        new_matrix = [[] for _ in range(len(self.vector))]
        for i in range(len(self.vector)):
            row = self.matrix.iloc[i].tolist()
            row = [abs(x) for x in row]
            max_element = max(row)
            max_el_index = row.index(max_element)
            if len(new_matrix[max_el_index]) != 0:
                return False
            new_matrix[max_el_index] = self.matrix.iloc[i]
            new_vector[max_el_index] = self.vector[i]
        self.vector = new_vector
        self.matrix = pd.DataFrame(data=new_matrix)
        return self.is_diag()

    def calculate_norm(self):
        parameters_sums = []
        for i in range(len(self.vector)):
            current_row_sum = 0
            a_ii = self.matrix.iloc[i,i]
            for j in range(len(self.vector)):
                if i == j:
                    continue
                current_row_sum += abs(self.matrix.iloc[i,j] / a_ii)
            parameters_sums.append(current_row_sum)
        self.__norm = max(parameters_sums)

    def solve(self):
        x_old = self.initial_value
        x_new = []
        iter_counter = 0
        precision_vector = []
        precision_not_satisfied = True
        while precision_not_satisfied:
            n = len(self.vector)

            for i in range(n):
                a_ii = self.matrix.iloc[i, i]
                x_i = self.vector[i] / a_ii

                for j in range(0, n):
                    if i == j:
                        continue
                    x_i -= self.matrix.iloc[i, j] * x_old[j] / a_ii 
                x_new.append(x_i)

            iter_counter += 1
            precision_vector = [abs(x_new[i] - x_old[i])
                                for i in range(len(x_old))]
            if max(precision_vector) < self.precision:
                precision_not_satisfied = False
            x_old = x_new
            x_new = []
        self.__solution = x_old
        self.__precision_vector = precision_vector
        self.__iter_counter = iter_counter

    @staticmethod
    def read_matrix_from_file(filename):
        matrix_rows = []
        try:
            with open(filename, 'r') as file:
                mat_dim = int(file.readline().strip())
                for _ in range(mat_dim):
                    matrix_rows.append([float(x)
                                       for x in file.readline().strip().split()])
                prec = float(file.readline().strip())
                if prec <= 0:
                    print("Precision must be greater than zero")
                    raise Exception
        except:
            print("An error occured during file reading. Check file permissons,filename and make sure matrix format is correct")
            raise SystemExit
        matrix, vector = LinearEquationSystem.array_to_matrix(matrix_rows)
        return matrix, vector, prec

    @staticmethod
    def ask_for_max_num():
        while True:
            try:
                inp = input("Enter max number's module: ").strip()
                max_num = float(inp)
                break
            except Exception as e:
                print(e)
                print("Max num must be a number")
        return max_num


    @staticmethod
    def array_to_matrix(matrix_rows):
        try:
            matrix = pd.DataFrame(data=matrix_rows)
            if matrix.shape[0] + 1 != matrix.shape[1]:
                raise Exception
        except:
            print(
                "Matrix has to be square, and B vector must be present as an additional column")
            raise SystemExit
        # Get the last column of matrix, which is B vector.
        vector = np.array(matrix.iloc[:, -1])
        # Remove it, so we're left with a square matrix
        matrix = matrix.iloc[:, :-1]
        return (matrix, vector)

    @staticmethod
    def read_matrix_from_console():
        matrix_rows = []
        while True:
            try:
                n = int(input("Please, enter matrix dimentionality: "))
                if n < 2:
                    print("Matrix dimentionality must be more or equal to 2")
                    raise Exception
                shall_generate_matrix = input("Would you like to generate matrix?(y/n): ")
                match shall_generate_matrix:
                    case "y":
                        max_num = LinearEquationSystem.ask_for_max_num()
                        matrix_rows = LinearEquationSystem.generate_matrix(n, max_num)
                    case _:
                        for _ in range(n):
                            matrix_rows.append([int(x) for x in input().split()])
                prec = float(input("Please, enter the precision: "))
                if prec <= 0:
                    print("Precision must be greater than zero")
                    raise Exception
                matrix, vector = LinearEquationSystem.array_to_matrix(
                    matrix_rows)
                return matrix, vector, prec
            except KeyboardInterrupt:
                raise SystemExit
            except:
                print("Error encountered, try to check if your data is valid.")
                matrix_rows = []


def main():
    from_file = input("Would you like to get matrix from file?(y/n): ")
    match from_file:
        case "y":
            filename = input("Type the filename(with extention): ")
            system = LinearEquationSystem(
                *LinearEquationSystem.read_matrix_from_file(filename))
        case _:
            system = LinearEquationSystem(
                *LinearEquationSystem.read_matrix_from_console())
    print("Norm is: ", system.norm)
    print("X vector is: ",system.solution)
    print("Precision vector: ", system.precision_vector)
    print("Iteration count is: ", system.iter_counter)


if __name__ == "__main__":
    main()
