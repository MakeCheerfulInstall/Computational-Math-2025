from itertools import permutations

import numpy as np

class SysOfLinEq:
    def __init__(self, file_name=''):
        self.answ = []
        self.variance = float('inf')
        self.iter_cnt = 0

        if not file_name:
            self.__input_params()
        else:
            self.__read_params(file_name)

        self.__is_epsilon_correct()

        assert self.__is_solvable(), "The condition of predominance of diagonal elements is not fulfilled"

        self.__create_coeff_matrix()
        self.__solution()

    def get_variance(self):
        return self.variance

    def get_iteration_count(self):
        return self.iter_cnt

    def get_matrix_norm(self):
        return np.max(np.sum(np.abs(self.KMatrix[:, :-1]), axis=1))

    def get_solution(self):
        return self.solution

    # решение СЛАУ
    def __solution(self):
        """
        У нас есть расширенная матрица со свободными коэфицентами
        тогда чтобы они имели вес ровно 1, добавляем единичный столбец
        """
        self.solution = np.concat(( self.KMatrix[:, self.dims], [1] ))

        while np.max(self.variance) >= self.epsilon:
            self.iter_cnt+=1
            old_solution = np.copy(self.solution)

            for line in range(self.dims):
                self.solution[line] = np.sum(self.KMatrix[line] * self.solution)

            self.variance = np.abs(self.solution - old_solution)
            
        self.solution = self.solution[:-1]
        self.variance = self.variance[:-1]

    # создание матрицы коэфициентов
    def __create_coeff_matrix(self):
        self.KMatrix = np.empty((self.dims, self.dims+1))

        for line in range(self.dims):
            self.KMatrix[line] = np.concat(( -self.matrix[line], [self.answ[line]]))
            self.KMatrix[line] /= self.matrix[line][line]
            self.KMatrix[line, line] = 0


    # выполняется ли условие сходимости
    def __is_solvable(self):
        # проверка, что вообще есть число большее суммы остальных в строке
        if self.__is_not_diagonally_dominant(np.abs(self.matrix)):
            return False

        # попытка поставить на диагональ максимальные числа
        for line in range(self.dims):
            sorted_indices = np.argsort(self.matrix[line:][:, line])[::-1]
            self.matrix[line:] = self.matrix[line:][sorted_indices]

        if self.__is_diagonally_dominant(np.abs(self.matrix)):
            return True
        
        # можно переставновками проверить, но это дольше
        # чем две верхних проверки
        for perm in permutations(range(self.dims)):
            permuted_matrix = self.matrix[list(perm), :]

            if self.__is_diagonally_dominant(np.abs(permuted_matrix)):
                self.matrix = permuted_matrix
                return True

        return False

    # проверка условия преобладания диагональных элементов
    def __is_not_diagonally_dominant(self, matrix):
        return np.any( (np.sum(matrix, axis=1) - 2*np.max(matrix, axis=1)) > 0 )

    # проверка условия преобладания диагональных элементов
    def __is_diagonally_dominant(self, matrix):
        one_strict = False
        diagonally_dominant = np.sum(matrix, axis=1) - 2*np.diag(matrix)

        if np.any(diagonally_dominant > 0):
            return False

        if np.any(diagonally_dominant < 0):
            one_strict = True

        return True & one_strict
    

    def __is_epsilon_correct(self):
        if self.epsilon < 0:
            raise ValueError("e must be positive")
    
    #  ввод всех параметров
    def __input_params(self):
        
        print("Input n: ", end="")
        self.dims = self.__get_param("n", int, input)
        print("Input e: ", end="")
        self.epsilon = self.__get_param("e", float, input)
        
        self.matrix = np.empty((self.dims, self.dims))
        
        print("\nInput matrix:")

        for line_cnt in range(self.dims):
            line_input = input().split()
            self.answ.append(line_input[-1])
            line = np.array(list(map(float, line_input[:-1])), dtype=np.float64)
            assert line.shape == (self.dims,), "Incorrect line shape"
            self.matrix[line_cnt] = line  

        print()   

    # чтение всех параметов из файла
    def __read_params(self, file_name):
        with open(file_name) as f:

            self.dims = self.__get_param("n", int, f.readline)
            self.epsilon = self.__get_param("e", float, f.readline)

            
            self.matrix = np.empty((self.dims, self.dims))

            for line_cnt in range(self.dims):
                line_file = f.readline().split()
                self.answ.append(line_file[-1])
                line = np.array(list(map(float, line_file[:-1])), dtype=np.float64)
                assert line.shape == (self.dims,), "Incorrect line shape"
                self.matrix[line_cnt] = line

            self.answ = np.array(self.answ, dtype=np.float64)

    # получение 1 параметра
    def __get_param(self, name, type, func):
        try:
            return type(func())
        except:
            raise ValueError(f"Incorrect {name}")  