using System;
using System.Windows;
using OxyPlot;
using OxyPlot.Series;
using CM1Lab.View;

namespace CM1Lab
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = this; // Устанавливаем контекст данных
            //MainWindow mainWindow = new MainWindow();
            this.WindowState = WindowState.Maximized;
        }

        private void gauss_seidelWindow_Click(object sender, RoutedEventArgs e)
        {
            Gauss_Seidel_MethodWindow gauss_seidelWindow = new Gauss_Seidel_MethodWindow();
            gauss_seidelWindow.Show();
            this.Close();

        }
    }
}
