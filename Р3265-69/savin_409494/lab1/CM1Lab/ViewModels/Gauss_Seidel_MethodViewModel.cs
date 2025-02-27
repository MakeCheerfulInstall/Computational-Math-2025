using OxyPlot;
using OxyPlot.Series;
using System.ComponentModel;
using CM1Lab.View;
using CM1Lab.ViewModels;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows;
using OxyPlot.Axes;
using System.IO;

namespace CM1Lab.ViewModels
{
    public class GaussSeidelViewModel : INotifyPropertyChanged
    {
        private string? accuracy;
        private string? size;
        private string? maxCountOfIter;
        private string? norma;
        private string? vector;
        private string? countOfIter;
        private string? vectorPogr;
        private PlotModel? plotModel;

        public event PropertyChangedEventHandler? PropertyChanged;

        private ObservableCollection<CoefficientsModel> coefficients = new ObservableCollection<CoefficientsModel>();
        private ObservableCollection<string> rightVector = new ObservableCollection<string>();
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public ObservableCollection<CoefficientsModel> Coefficients
        {
            get => coefficients;
            set { coefficients = value; OnPropertyChanged(nameof(Coefficients)); }
        }

        public ObservableCollection<string> RightVector
        {
            get => rightVector;
            set { rightVector = value; OnPropertyChanged(nameof(RightVector)); }
        }
        public PlotModel PlotModel
        {
            get => plotModel;
            set
            {
                plotModel = value;
                OnPropertyChanged(nameof(PlotModel)); // Уведомляем UI
            }
        }
        public string Accuracy
        {
            get => accuracy;
            set { accuracy = value; OnPropertyChanged(nameof(Accuracy)); }
        }

        public string Size
        {
            get => size;
            set { size = value; OnPropertyChanged(nameof(Size)); }
        }

        public string MaxCountOfIter
        {
            get => maxCountOfIter;
            set { maxCountOfIter = value; OnPropertyChanged(nameof(MaxCountOfIter)); }
        }

        public string Norma
        {
            get => norma;
            set { norma = value; OnPropertyChanged(nameof(Norma)); }
        }

        public string Vector
        {
            get => vector;
            set { vector = value; OnPropertyChanged(nameof(Vector)); }
        }

        public string CountOfIter
        {
            get => countOfIter;
            set { countOfIter = value; OnPropertyChanged(nameof(CountOfIter)); }
        }

        public string VectorPogr
        {
            get => vectorPogr;
            set { vectorPogr = value; OnPropertyChanged(nameof(VectorPogr)); }
        }

        private double[,] A;
        private double[] b;

        // Метод решения системы линейных уравнений методом Гаусса-Зейделя
        public (double[] solution, int iterations) Gauss_Seidel_Count()
        {
            // Читаем размерность матрицы
            int n = int.Parse(Size);
            // Читаем точность вычислений
            double epsilon = double.Parse(Accuracy);
            // Читаем максимальное количество итераций
            int M = int.Parse(MaxCountOfIter);

            // Инициализируем матрицу коэффициентов A
            A = new double[n, n];
            // Инициализируем вектор правой части b
            b = new double[n];

            // Заполняем вектор b значениями из RightVector
            for (int i = 0; i < n; i++)
            {
                b[i] = double.Parse(RightVector[i]);
            }

            // Заполняем матрицу A из списка Coefficients
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int index = i * n + j; // Преобразуем индексы двумерного массива в одномерный
                    A[i, j] = (index < Coefficients.Count) ? Coefficients[index].CoeffA : 0;
                }
            }

            // Проверяем наличие диагонального преобладания
            if (!CheckDiagonalDominance(A))
            {
                MessageBox.Show("Матрица не имеет диагонального преобладания. Попытка перестановки строк...");
                // Если невозможно достичь диагонального преобладания, завершаем выполнение
                if (!MakeDiagonalDominance(A, b))
                {
                    MessageBox.Show("Невозможно достичь диагонального преобладания.");
                    //return (null, 0);
                }
            }

            // Преобразуем систему к виду x = Cx + d
            double[,] C = new double[n, n];
            double[] d = new double[n];

            for (int i = 0; i < n; i++)
            {
                // Вычисляем вектор d
                d[i] = b[i] / A[i, i];
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                        C[i, j] = 0; // Диагональные элементы C равны нулю
                    else
                        C[i, j] = -A[i, j] / A[i, i]; // Остальные элементы C
                }
            }

            // foreach (var item in A)
            // {
            //     Debug.WriteLine(item);   
            // }

            // foreach (var item in b)
            // {
            //     Debug.WriteLine(item);
            // }

            // Вычисляем норму матрицы C
            double normC = CalculateMatrixNorm(C);
            Debug.WriteLine($"Норма матрицы C: {normC}");
            Norma = normC.ToString();

            // Инициализируем начальное приближение
            double[] x = new double[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = b[i] / A[i, i];
            }

            // Создаем массив для хранения предыдущих значений x
            double[] prevX = new double[n];
            Array.Copy(x, prevX, n);

            int k; // Переменная для отслеживания номера текущей итерации
            for (k = 0; k < M; k++) // Итерационный процесс: ограничен максимальным числом итераций M
            {
                // Проходим по каждому элементу вектора x
                for (int i = 0; i < n; i++) 
                {
                    double sum = 0; // Переменная для хранения суммы произведений элементов матрицы C и значений вектора x
                    // Вычисляем сумму произведений элементов строки матрицы C и значений вектора
                    for (int j = 0; j < n; j++) 
                    {
                        // Суммируем произведение элементов матрицы C и соответствующих элементов вектора x
                        // Если j < i, берем значение из текущего вектора x, иначе используем значение из предыдущего вектора prevX
                        sum += C[i, j] * (j < i ? x[j] : prevX[j]); 
                    }
                    // Обновляем значение x[i], добавляя d[i] и сумму произведений
                    x[i] = d[i] + sum;
                }

                // Создаем массив для вычисления погрешности
                double[] error = new double[n]; 
                // Вычисляем погрешности для каждого элемента вектора
                for (int i = 0; i < n; i++) 
                {
                    // Погрешность для каждого элемента — это разница между текущим и предыдущим значением x
                    error[i] = Math.Abs(x[i] - prevX[i]);
                }

                // Выводим результаты текущей итерации в отладочный вывод (Debug)
                Debug.WriteLine($"Итерация {k + 1}:");
                Debug.WriteLine($"x1 = {x[0]}, x2 = {x[1]}, x3 = {x[2]}"); 
                Debug.WriteLine($"Погрешности: e1 = {error[0]}, e2 = {error[1]}, e3 = {error[2]}");

                // Записываем текущие значения вектора и погрешности для отображения в пользовательском интерфейсе
                Vector = $"x1 = {x[0]}, x2 = {x[1]}, x3 = {x[2]}"; 
                CountOfIter = (k + 1).ToString(); // Записываем количество выполненных итераций
                VectorPogr = $"e1 = {error[0]}, e2 = {error[1]}, e3 = {error[2]}"; // Записываем погрешности для UI

                // Проверка сходимости: если максимальная погрешность меньше заданного порога (epsilon), выходим из цикла
                double maxDiff = error.Max(); // Находим максимальную погрешность
                if (maxDiff < epsilon) // Если максимальная погрешность меньше порога
                    break; // Выход из цикла, так как сходимость достигнута

                // Обновляем prevX для следующей итерации, копируя текущие значения x
                Array.Copy(x, prevX, n); // Копируем элементы из x в prevX для использования на следующей итерации
            }

            // Возвращаем результат: конечный вектор x и количество итераций
            return (x, k + 1);

        }


        // Метод проверки диагонального преобладания матрицы
        private bool CheckDiagonalDominance(double[,] A)
        {
            int n = A.GetLength(0); // Получаем размерность матрицы (количество строк)
            for (int i = 0; i < n; i++) // Проходим по каждой строке
            {
                double sum = 0; // Инициализируем сумму элементов строки (кроме диагонального)
                for (int j = 0; j < n; j++) // Проходим по каждому элементу в строке
                {
                    if (i != j) // Пропускаем диагональный элемент
                        sum += Math.Abs(A[i, j]); // Добавляем модуль элемента к сумме
                }
                if (Math.Abs(A[i, i]) <= sum) // Проверяем условие диагонального преобладания
                    return false; // Если нарушено, возвращаем false
            }
            return true; // Если проверка пройдена для всех строк, возвращаем true
        }

        // Метод перестановки строк для достижения диагонального преобладания
        private bool MakeDiagonalDominance(double[,] A, double[] b)
        {
            int n = A.GetLength(0); // Получаем размерность матрицы
            bool[] usedRows = new bool[n]; // Создаём массив для отслеживания использованных строк
            double[,] newA = new double[n, n]; // Создаём новую матрицу для переставленных строк
            double[] newB = new double[n]; // Создаём новый вектор b

            for (int i = 0; i < n; i++) // Проходим по каждой строке
            {
                int bestRow = -1; // Переменная для хранения индекса строки с наибольшим диагональным элементом
                double maxDiagonal = 0; // Переменная для хранения максимального диагонального элемента

                for (int j = 0; j < n; j++) // Ищем строку с наибольшим диагональным элементом
                {
                    if (!usedRows[j] && Math.Abs(A[j, i]) > maxDiagonal) // Проверяем, не использована ли строка и является ли диагональный элемент максимальным
                    {
                        maxDiagonal = Math.Abs(A[j, i]); // Обновляем максимальный диагональный элемент
                        bestRow = j; // Запоминаем индекс строки
                    }
                }

                if (bestRow == -1 || maxDiagonal == 0) // Если не удалось найти подходящую строку
                    return false; // Возвращаем false (невозможно достичь диагонального преобладания)

                usedRows[bestRow] = true; // Помечаем строку как использованную

                for (int j = 0; j < n; j++) // Копируем строку в новую матрицу
                    newA[i, j] = A[bestRow, j];

                newB[i] = b[bestRow]; // Копируем соответствующий элемент вектора b
            }

            Array.Copy(newA, A, A.Length); // Копируем переставленную матрицу обратно в A
            Array.Copy(newB, b, b.Length); // Копируем переставленный вектор обратно в b

            return CheckDiagonalDominance(A); // Проверяем, достигнуто ли диагональное преобладание
        }

        // Метод вычисления нормы матрицы (максимальной суммы элементов в строках)
        private double CalculateMatrixNorm(double[,] A)
        {
            int n = A.GetLength(0); // Получаем размерность матрицы
            double norm = 0; // Инициализируем переменную для хранения нормы
            for (int i = 0; i < n; i++) // Проходим по каждой строке
            {
                double rowSum = 0; // Переменная для хранения суммы элементов строки
                for (int j = 0; j < n; j++) // Считаем сумму модулей элементов строки
                {
                    rowSum += Math.Abs(A[i, j]);
                }
                norm = Math.Max(norm, rowSum); // Обновляем норму, если текущая сумма больше
            }
            return norm; // Возвращаем вычисленную норму
        }



        public void BuildGraphic()
        {
            var model = new PlotModel { Title = "Сходимость метода Гаусса-Зейделя" };

            // Создаем три серии для x1, x2, x3
            var seriesX1 = new LineSeries { Title = "x1", Color = OxyColors.Blue };
            var seriesX2 = new LineSeries { Title = "x2", Color = OxyColors.Red };
            var seriesX3 = new LineSeries { Title = "x3", Color = OxyColors.Green };

            // Решаем систему методом Гаусса-Зейделя
            var (solution, iterations) = Gauss_Seidel_Count();

            // Если решение найдено, строим график
            if (solution != null)
            {
                // Инициализируем начальные приближения
                double[] x = new double[solution.Length];
                x[0] = 1.2; // Начальное приближение для x1
                x[1] = 1.3; // Начальное приближение для x2
                x[2] = 1.4; // Начальное приближение для x3

                double[] prevX = new double[solution.Length];
                Array.Copy(x, prevX, solution.Length);

                int M = int.Parse(MaxCountOfIter);
                double epsilon = double.Parse(Accuracy);

                for (int k = 0; k < M; k++)
                {
                    // Выполняем одну итерацию метода Гаусса-Зейделя
                    for (int i = 0; i < solution.Length; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < solution.Length; j++)
                        {
                            if (i != j)
                                sum += -A[i, j] / A[i, i] * (j < i ? x[j] : prevX[j]);
                        }
                        x[i] = b[i] / A[i, i] + sum;
                    }

                    // Добавляем точки на график
                    seriesX1.Points.Add(new DataPoint(k, x[0]));
                    seriesX2.Points.Add(new DataPoint(k, x[1]));
                    seriesX3.Points.Add(new DataPoint(k, x[2]));

                    // Проверка сходимости
                    double maxDiff = 0;
                    for (int i = 0; i < solution.Length; i++)
                    {
                        maxDiff = Math.Max(maxDiff, Math.Abs(x[i] - prevX[i]));
                    }

                    if (maxDiff < epsilon)
                        break;

                    // Обновляем prevX для следующей итерации
                    Array.Copy(x, prevX, solution.Length);
                }
            }

            // Добавляем серии на график
            model.Series.Add(seriesX1);
            model.Series.Add(seriesX2);
            model.Series.Add(seriesX3);

            // Устанавливаем оси
            model.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Итерация" });
            model.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Значение x" });

            // Устанавливаем модель графика
            PlotModel = model;
        }

        public void OnSizeChanged()
        {
            int size = int.Parse(Size);
            Coefficients.Clear();
            RightVector.Clear();

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    Coefficients.Add(new CoefficientsModel { CoeffA = 1 }); // Значение по умолчанию
                }
                RightVector.Add("1"); // Значение правой части по умолчанию
            }
        }

        public void LoadDataFromText(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException("Файл не найден.");
            }

            var lines = File.ReadAllLines(filePath);

            foreach (var line in lines)
            {
                var parts = line.Split(new[] { ' ' }, 2); // Разделяем строку на ключ и значение
                if (parts.Length < 2)
                    continue;

                var key = parts[0].Trim();
                var value = parts[1].Trim();

                switch (key)
                {
                    case "Точность":
                        Accuracy = value;
                        break;
                    case "Размерность":
                        Size = value;
                        break;
                    case "МаксКоличестоИтераций":
                        MaxCountOfIter = value;
                        break;
                    case "Коэффициенты":
                        Coefficients.Clear();
                        var coeffs = value.Split(' ').Select(double.Parse).ToList();
                        foreach (var coeff in coeffs)
                        {
                            Coefficients.Add(new CoefficientsModel { CoeffA = coeff });
                        }
                        break;
                    case "Векторы":
                        RightVector.Clear();
                        var vectors = value.Split(' ').Select(double.Parse).ToList();
                        foreach (var vector in vectors)
                        {
                            RightVector.Add(vector.ToString());
                        }
                        break;
                }
            }
        }
    }
}