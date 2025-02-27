using Microsoft.Win32;
using OxyPlot.Series;
using OxyPlot;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using CM1Lab.ViewModels;
using System.Windows.Data;

namespace CM1Lab.View
{
    public partial class Gauss_Seidel_MethodWindow : Window
    {
        private GaussSeidelViewModel vm; // Переменная для ViewModel

        public Gauss_Seidel_MethodWindow()
        {
            InitializeComponent();
            vm = new GaussSeidelViewModel();
            DataContext = vm; // Устанавливаем ViewModel в качестве DataContext
            this.WindowState = WindowState.Maximized;
        }

        public void ChooseWayClick(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Текстовые файлы (*.txt)|*.txt|Все файлы (*.*)|*.*";
            openFileDialog.Title = "Выберите файл с данными";

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    vm.LoadDataFromText(openFileDialog.FileName);
                    MessageBox.Show("Данные успешно загружены из файла.", "Успех", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка при загрузке данных из файла: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }



        public void CountResults(object sender, EventArgs e)
        {
            vm.BuildGraphic(); // Теперь вызываем метод из ViewModel
            vm.Gauss_Seidel_Count();
        }

        public void ConfirmClick(object sender, EventArgs e)
        {
            //vm.BuildGraphic(); // Теперь вызываем метод из ViewModel
            //vm.Gauss_Seidel_Count();
            vm.OnSizeChanged();
            UpdateCoefficientGrid();
        }

        public void UpdateCoefficientGrid()
        {
            int size = int.Parse(vm.Size);
            CoefficientGrid.Children.Clear();
            CoefficientGrid.RowDefinitions.Clear();
            CoefficientGrid.ColumnDefinitions.Clear();

            double cellWidth = Math.Max(200.0 / size, 20);
            double cellHeight = 20;

            for (int i = 0; i < size; i++)
            {
                CoefficientGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

                for (int j = 0; j < size; j++)
                {
                    if (i == 0)
                    {
                        CoefficientGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                    }

                    TextBox textBox = new TextBox
                    {
                        Width = cellWidth,
                        Height = cellHeight,
                        Margin = new Thickness(1)
                    };

                    // Привязка TextBox к CoeffA
                    Binding binding = new Binding($"Coefficients[{i * size + j}].CoeffA")
                    {
                        Source = vm,
                        Mode = BindingMode.TwoWay,
                        UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
                    };

                    textBox.SetBinding(TextBox.TextProperty, binding);

                    Grid.SetRow(textBox, i);
                    Grid.SetColumn(textBox, j);
                    CoefficientGrid.Children.Add(textBox);
                }

                // Добавляем столбец для вектора правой части (RightVector)
                if (i == 0)
                {
                    CoefficientGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                }

                TextBox rightVectorTextBox = new TextBox
                {
                    Width = cellWidth,
                    Height = cellHeight,
                    Margin = new Thickness(1)
                };

                // Привязка TextBox к RightVector
                Binding rightBinding = new Binding($"RightVector[{i}]")
                {
                    Source = vm,
                    Mode = BindingMode.TwoWay,
                    UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
                };

                rightVectorTextBox.SetBinding(TextBox.TextProperty, rightBinding);

                Grid.SetRow(rightVectorTextBox, i);
                Grid.SetColumn(rightVectorTextBox, size); // Колонка после коэффициентов
                CoefficientGrid.Children.Add(rightVectorTextBox);
            }
        }



    }
}
