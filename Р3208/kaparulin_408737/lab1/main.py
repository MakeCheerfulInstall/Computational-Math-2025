from SysOfLinEq import SysOfLinEq





if __name__ == "__main__":
    sle = SysOfLinEq("pars.txt") # pars.txt

    print("Решение: ", sle.get_solution())
    print("Норма матрицы: ", sle.get_matrix_norm())
    print("Кол-во итераций: ", sle.get_iteration_count())
    print("Вектор погрешностей: ", sle.get_variance())
