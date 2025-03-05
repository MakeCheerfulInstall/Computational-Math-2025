package main

// Determinant вычисляет определитель матрицы любого размера методом Гаусса
func Determinant(a [][]float64) (float64, [][]float64) {
	n := len(a)
	m := make([][]float64, n)
	for i := range a {
		m[i] = append([]float64{}, a[i]...)
	}

	det := 1.0
	for i := 0; i < n; i++ {
		// перестановка строк, если на главной диагонали 0
		if m[i][i] == 0 {
			for k := i + 1; k < n; k++ {
				if m[k][i] != 0 {
					m[i], m[k] = m[k], m[i]
					det *= -1
					break
				}
			}
		}
		if m[i][i] == 0 {
			return 0, nil
		}

		det *= m[i][i]
		// прямой ход метода гаусса
		GaussSolverForward(i, m)
	}
	return det, m
}

func GaussSolverForward(i int, m [][]float64) {
	n := len(m)
	for k := i + 1; k < n; k++ {

		factor := m[k][i] / m[i][i]
		for j := i; j < n; j++ {
			m[k][j] -= factor * m[i][j]
		}
		m[k][n] -= factor * m[i][n]
	}
}

func GaussSolverBackward(m [][]float64) []float64 {
	n := len(m)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {

		sum := m[i][n]
		for j := i + 1; j < n; j++ {
			sum -= m[i][j] * x[j]
		}
		x[i] = sum / m[i][i]
	}
	return x
}

func CalculateDeltas(a [][]float64, x []float64) []float64 {
	n := len(a)
	deltas := make([]float64, n)

	for i := 0; i < n; i++ {
		left := 0.0
		right := a[i][n]

		for j := 0; j < n; j++ {
			left += a[i][j] * x[j]
		}
		deltas[i] = right - left
	}
	return deltas
}
