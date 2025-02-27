using OxyPlot;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CM1Lab.ViewModels
{
    public class CoefficientsModel : INotifyPropertyChanged
    {
        private double coeffA;
        private double coeffB;
        private string? xi;

        public double CoeffA
        {
            get => coeffA;
            set { coeffA = value; OnPropertyChanged(nameof(CoeffA)); }
        }

        public double CoeffB
        {
            get => coeffB;
            set { coeffB = value; OnPropertyChanged(nameof(CoeffB)); }
        }

        public string? Xi
        {
            get => xi;
            set { xi = value; OnPropertyChanged(nameof(Xi)); }
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
