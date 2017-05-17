using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace task3_console
{
    public class neuron
    {
        public List<double> w = new List<double> { };
        public List<double> w_prev = new List<double> { };
        public double y {get; set;}
        public double err {get; set;}
        public double e;    //взвешенная сумма входных сигналов
        public double delta;

        public double res()
        {
            return y;
        }
        public neuron(int prev_layer_size, Random rnd)
        {
            y = 0;
            for (int i = 0; i < prev_layer_size; ++i)
            {
                w.Add(rnd.Next(-50, 50) / 100.0 + i/100);
               // w_prev.Add(0);
            }
        }
        public neuron(List<double> weights)
        {
            y = 0;
            for (int i = 0; i < weights.Count; ++i)
            {
                w.Add(weights[i]);
               // w_prev.Add(0);
            }
        }
        public void neuron_activation(List<double> y_n, Func<int, double, double> func, int num_func)
        {
            y = 0;
            double sum = 0;
            for(int i = 0; i < y_n.Count; ++i)
                sum += y_n[i] * w[i];
            e = sum;
            y = func(num_func, sum);
        }
    }
}
