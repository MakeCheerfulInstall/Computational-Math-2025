package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func CallUI() {
	fmt.Println("How do you want to enter the matrix?\nType \"f\" for file or \"k\" for keyboard" +
		" or \"r\" to generate random matrix and write it to file")
	var ans string
	fmt.Fscanln(os.Stdin, &ans)

	var matrix [][]float64

	switch ans {
	case "f":
		var err error
		var path string

		path, err = readFilePath()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

		matrix, err = ReadMatrixFromFile(path)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	case "k":
		var err error

		matrix, err = ReadMatrixFromKeyboard()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	case "r":
		//var err error
		path, err := GenerateRandomMatrixFile()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		matrix, err = ReadMatrixFromFile(path)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

	default:
		fmt.Println("idk what it means :(")
		os.Exit(1)
	}

	//fmt.Println(matrix)
	//os.Exit(1)

	// Вычисление и вывод определителя
	det, m := Determinant(matrix)
	fmt.Println("Определитель матрицы:", det, "\nМатрица: ", m)

	// обратный ход метода гаусса
	x := GaussSolverBackward(m)
	fmt.Println("Решения: ", x)

	deltas := CalculateDeltas(matrix, x)
	fmt.Println("Невязки: ", deltas)
}

func makeEmptyMatrix(n int) [][]float64 {
	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n+1)
	}
	return matrix
}

func checkDimensions(n int) error {
	if n < 2 || n > 20 {
		return fmt.Errorf("unsupported dimensions: %v", n)
	}
	return nil
}

func ReadMatrixFromKeyboard() ([][]float64, error) {
	n := 0
	fmt.Println("Type n - matrix dimension from 2 to 20 (only the first number is read):")

	_, err := fmt.Fscan(os.Stdin, &n)
	if err != nil {
		return nil, fmt.Errorf("error reading dimensions: %v", err)
	}
	dimErr := checkDimensions(n)
	if dimErr != nil {
		return nil, dimErr
	}

	matrix := makeEmptyMatrix(n)

	fmt.Println("Enter matrix in format ")
	fmt.Println("a_11")
	fmt.Println("a_12")
	fmt.Println("....")
	fmt.Println("a_1n")
	fmt.Println("b_1")
	fmt.Println("....")
	fmt.Println("a_k1")
	fmt.Println("a_k2")
	fmt.Println("....")
	fmt.Println("a_kn")
	fmt.Println("b_k")

	for i := 0; i < n; i++ {
		for j := 0; j < n+1; j++ {
			var inputNum float64

			_, err := fmt.Fscan(os.Stdin, &inputNum)
			if err != nil {
				return nil, fmt.Errorf("error reading matrix: %v", err)
			}

			matrix[i][j] = inputNum
		}
	}

	return matrix, nil
}

func ReadMatrixFromFile(path string) ([][]float64, error) {
	file, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}
	strings.TrimSpace(string(file))

	rows := strings.Split(string(file), "\n")
	for i, row := range rows {
		if len(row) == 0 {
			rows = append(rows[:i], rows[i+1:]...)
		}
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("file is empty: %v", err)
	}

	n, err := strconv.Atoi(strings.TrimSpace(rows[0]))
	if err != nil {
		return nil, fmt.Errorf("error converting matrix dimension: %v", err)
	}
	dimErr := checkDimensions(n)
	if dimErr != nil {
		return nil, dimErr
	}

	if len(rows)-1 < 2 || len(rows)-1 > 20 {
		return nil, fmt.Errorf("unsupported matrix dimension")
	}

	for i := 1; i < n; i++ {
		rows[i] = strings.TrimSpace(rows[i])
		if len(strings.Split(rows[i], " ")) != n+1 {
			return nil, fmt.Errorf("matrix isn't square or you forgot right side of equations or its empty")
		}
	}

	if len(rows)-1 != n {
		return nil, fmt.Errorf("number of matrix rows doesn't match dimension")
	}

	matrix := makeEmptyMatrix(n)

	for i := 0; i < n; i++ {
		rowNums := strings.Split(rows[i+1], " ")
		for j := 0; j < n+1; j++ {
			num, err := strconv.ParseFloat(rowNums[j], 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing matrix number: %v", err)
			}
			matrix[i][j] = num
		}
	}

	return matrix, nil
}

func readFilePath() (string, error) {
	fmt.Println("Enter valid path to file:")
	var path string
	_, err := fmt.Fscan(os.Stdin, &path)

	if err != nil {
		return "", fmt.Errorf("error reading file path: %v", err)
	}
	return path, nil
}

func GenerateRandomMatrixFile() (string, error) {
	n := 0
	fmt.Println("Type n - matrix dimension from 2 to 20 (only the first number is read):")

	_, err := fmt.Fscan(os.Stdin, &n)
	if err != nil {
		return "", fmt.Errorf("error reading dimensions: %v", err)
	}
	dimErr := checkDimensions(n)
	if dimErr != nil {
		return "", dimErr
	}

	//matrix := makeEmptyMatrix(n)
	var rnd = rand.New(rand.NewSource(time.Now().UnixNano()))

	file, err := os.Create("matrix.txt")
	if err != nil {
		return "", fmt.Errorf("error creating file: %v", err)
	}
	// close fo on exit and check for its returned error
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("error closing file: %v", err)
		}
	}()

	var buffer bytes.Buffer
	buffer.Write([]byte(strconv.Itoa(n) + "\n"))
	for i := 0; i < n; i++ {
		for j := 0; j < n+1; j++ {
			// min + rand.Float64() * (max - min)
			buffer.Write([]byte(strconv.FormatFloat((-15)+rnd.Float64()*(15-(-15)), 'f', 4, 64) + " "))
			//matrix[i][j] = (-15) + rnd.Float64()*(15-(-15))
		}
		buffer.Write([]byte("\n"))
	}

	if _, err := file.Write(buffer.Bytes()); err != nil {
		return "", fmt.Errorf("error writing to file: %v", err)
	}

	return file.Name(), nil
}
