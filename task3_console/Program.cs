using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.IO;
//using System.Windows.Media;
//using System.Windows.Media.Imaging;

/*TODO:
     * 1)Изменение скорости обучения
     * 2)Оставшуюся часть выборки поместить в часть выборки для проверки
     * 3)Организовать несколько критериев останова
 *          -изменение между весами на разных эпохах не происходит(сумму взять)
 *          -разница между среднеквадратической ошибкой за 3 эпохи не значительна
 *          -максимальная итерация
 *     4)выбор файла с весами
     */
namespace task3_console
{
    class Program
    {
        static public List<LetterImage> process_input_images(int start, int num_images, string location)
        {
            List<LetterImage> images = new List<LetterImage> { };
            try
            {
                //using (StreamReader str = new StreamReader("C:/Users/Margo/Desktop/нейронные сети/train-images-idx3-ubyte/letter-recognition.txt", System.Text.Encoding.Default))
                using (StreamReader str = new StreamReader(location, Encoding.Default))
                {
                    for (int i = 0; i < start; ++i)
                        str.ReadLine();
                    for (int i = 0; i < num_images; ++i)
                    {
                        LetterImage elem = new LetterImage(str.ReadLine());
                        images.Add(elem);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
            return images;
        }
        //function fi for the forward computation
        static public double activationFunction(int num, double v)
        {
            if (num == 1)
            {
                double a = 5.3;   //a > 0
                return 1 / (1 + Math.Exp(-a * v));
            }
            else
                if (num == 2)
                {
                    // double a = 1.7159, b = 2 / 3.0;   //(a, b) > 0
                    double a = 0.5, b = 0.2;
                    return a * Math.Tanh(b * v);
                }
                else
                    if(num == 4)
                    {
                        double res = 0, F = 30000, alpha = 1 / 2.0;
                        if (v < 0)
                            res = 0;
                        else
                        {
                            if (v > F)
                                res = 1;
                            else
                                res = alpha * v;
                        }
                        return res;
                    }
                    else 
                        return 0;
        }
        static public double diff_activationFunction(int num, double v)
        {
            if (num == 1)
            {
                double a = 5.3;
                return a * Math.Exp(-a * v) / (Math.Pow((1 + Math.Exp(-a * v)), 2));
            }
            else
                if (num == 2)
                {
                    //double a = 1.7159, b = 2 / 3.0;    //(a, b) > 0
                    double a = 0.5, b = 0.2;
                    return a * b * (1 - Math.Pow(Math.Tanh(b * v), 2));
                }
                else return 0;
        }
        static public bool barrier_function(List<double> last_layer, double eps)
            {
                bool criteria = false;
                double sum = 0;
                for (int i = 0; i < last_layer.Count; ++i)
                {
                    sum += last_layer[i] * last_layer[i];
                }
                sum = Math.Sqrt(sum);
                if (sum < eps)
                    criteria = true;

                return criteria;
            }
        static void Main(string[] args)
        {
            Random rnd = new Random();
            Random rnd1 = new Random();

            string src = "C:/Users/Margo/Desktop/Mine/Study/4 course/2 term";
            string path = src + "/нейронные сети/task3_console/bin/Release/";
            string images_location = src + "/нейронные сети/train-images-idx3-ubyte/letter-recognition.txt";
            string file_with_weights = "";
            /*постоянная момента равна нулю сейчас, ввести ее*/
            double a_moment = 0.5; //0<|a|<1
            //numberOf samples
            int N;
            //скорость обучения
            double speed = 0;
            /*Скорость обучения:
            Правило 1. Каждый параметр сети (вес) - свое
            значение скорости обучения
            Правило 2. Должна меняться на итерации
            Правило 3. Увеличиваем, если на нескольких
            итерациях локальный градиент имеет один зн
            ак, иначе уменьшаем*/

            //критерий оценки среднеквадратической ошибки
            double stop_epoha_err = 0.01;

            //кол-во элементов в обучающей выборке
            int size_elems_by_epoch = 0;
            int stop_num_epoh = 0;
            int activ_func_num = 0;

            double last_err = 0;
            //length of img
            int len;

            //number of hidden layers (ml,  l = 0..L)
            int num_of_hid_layers = 0;
            int num_of_layers = 0;
            int num_of_output_neur = 26;

            //number of neurons
            int[] num_of_neurons_in_particular_layer = new int[] { };

            //x_n - входной вектор, предъявляемый входному слою сенсорных узлов
            List<double> x_n = new List<double> { };
            List<int> d_n = new List<int> { };
            List<double> v_n = new List<double> { };

            //yi^(l - 1) - выходной(функциональный) сигнал нейрона i на пред слое (l - 1)
            List<List<double>> y = new List<List<double>> { };

            //wij^(l) -синаптический вес связи нейрона j слоя l с нейроном слоя l - 1
            double[][] w_l_n = new double[][] { };
            double[][] w_prev = new double[][] { };
            List<List<neuron>> ner = new List<List<neuron>> { };
            List<double> err = new List<double> { };
            List<double> ner_input_y = new List<double> { };
            List<double> ner_res = new List<double> { };
            List<LetterImage> input_images = new List<LetterImage> { };
            List<LetterImage> check_images = new List<LetterImage> { };

            int num_process;// = 0;
            Console.Write("Create (0) or try (1) neuron net? -> ");
            num_process = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine(num_process.ToString());
            if (num_process == 0)
            {
                Console.Write("Do you want customise settings? Yes(0) No(1) ->");
                int ans1 = Convert.ToInt32(Console.ReadLine());
                if (ans1 == 1)
                {
                  /*  speed = 0.01;
                    //кол-во элементов в обучающей выборке
                    size_elems_by_epoch = 1000;
                    stop_num_epoh = 1;
                    activ_func_num = 1;

                    num_of_hid_layers = 2;
                    num_of_layers = num_of_hid_layers + 1; //+output layer
                    num_of_neurons_in_particular_layer = new int[num_of_layers];// 2;
                    num_of_neurons_in_particular_layer[0] = 50;
                    num_of_neurons_in_particular_layer[1] = 70;
                    num_of_neurons_in_particular_layer[2] = num_of_output_neur;*/

                    speed = 0.01;
                    //кол-во элементов в обучающей выборке
                    size_elems_by_epoch = 1;
                    stop_num_epoh = 1;
                    activ_func_num = 1;

                    num_of_hid_layers = 2;
                    num_of_layers = num_of_hid_layers + 1; //+output layer
                    num_of_neurons_in_particular_layer = new int[num_of_layers];// 2;
                    num_of_neurons_in_particular_layer[0] = 2;
                    num_of_neurons_in_particular_layer[1] = 2;
                    num_of_neurons_in_particular_layer[2] = num_of_output_neur;



                }
                else
                    if (ans1 == 0)
                    {
                        Console.WriteLine("1) Net settings:");
                        Console.Write("How many hidden layers in the net?->");
                        num_of_hid_layers = Convert.ToInt32(Console.ReadLine());
                        num_of_layers = num_of_hid_layers + 1; //+output layer
                        num_of_neurons_in_particular_layer = new int[num_of_layers];

                        for (int i = 0; i < num_of_hid_layers; ++i)
                        {
                            Console.Write("Number of neurons on the " + i + " layer = ");
                            num_of_neurons_in_particular_layer[i] = Convert.ToInt32(Console.ReadLine());
                        }
                        num_of_neurons_in_particular_layer[num_of_hid_layers] = num_of_output_neur;
                        Console.WriteLine("2) Train settings:");
                        Console.Write("How many input files? ->");
                        size_elems_by_epoch = Convert.ToInt32(Console.ReadLine());
                        Console.Write("Speed of net(use \',' as a delimeter of double)? ->");
                        speed = Convert.ToDouble(Console.ReadLine());
                        Console.Write("Number of activation function(1 - logic, 2 - tanh) ->");
                        activ_func_num = Convert.ToInt32(Console.ReadLine());
                        Console.Write("Maximum number of iterations = ");
                        stop_num_epoh = Convert.ToInt32(Console.ReadLine());
                    }
               // stop_epoha_err = 0.25;

                /*Создание списка изображений букв*/
                input_images = process_input_images(0, size_elems_by_epoch, images_location);//считываем с начала файла, поэтому 0
                N = input_images[0].components.Count;
                len = N;

                int curr_size = N;
                //создание сети
                for (int l = 0; l < num_of_layers; ++l) //+1
                {
                    List<neuron> ner_l = new List<neuron>();
                    for (int j = 0; j < num_of_neurons_in_particular_layer[l]; ++j)
                    {
                        neuron n = new neuron(curr_size, rnd);
                        ner_l.Add(n);
                    }
                    ner.Add(ner_l);
                    curr_size = num_of_neurons_in_particular_layer[l];
                }

                bool end_criteria = false;
                int epoha_count = 0;
                double prev_epoha_mid_err_1 = -200, prev_epoha_mid_err_2 = -100, prev_sum_weights = 0, sum_weights_for_epoha = 0;
                //цикл по эпохам
                while (!end_criteria)
                {
                    List<double> images_err = new List<double> { };
                    //цикл по изображениям
                    for (int p = 0; p < size_elems_by_epoch; ++p)
                    {
                        d_n = input_images[p].d_n;

                        //прямой ход
                        ner_input_y.Clear();
                        for (int j = 0; j < N; ++j)
                            ner_input_y.Add(input_images[p].components[j] / 15); //  /15 - нормировка

                        for (int l = 0; l < num_of_layers; ++l) //+1
                        {
                            for (int j = 0; j < num_of_neurons_in_particular_layer[l]; ++j)
                            {
                                ner[l][j].neuron_activation(ner_input_y, activationFunction, activ_func_num);
                                ner_res.Add(ner[l][j].y);
                            }
                            ner_input_y.Clear();
                            for (int i = 0; i < ner_res.Count; ++i)
                                ner_input_y.Add(ner_res[i]);
                            ner_res.Clear();
                        }

                        //чисто для отладки
          //              Console.WriteLine("\n-> WEIGHTS \n");

          /*              for (int i1 = 0; i1 < num_of_layers; ++i1)
                        {
                            string s1 = "";
                            for (int j1 = 0; j1 < num_of_neurons_in_particular_layer[i1]; ++j1)
                                for(int k1 = 0; k1 < ner[i1][j1].w.Count; ++k1)
                                s1 += ner[i1][j1].w[k1].ToString() + "  ,  ";
                            Console.WriteLine(s1 + "\n");
                        }//*/
                        
                        //чисто для отладки

         //               Console.WriteLine("-> ERR (last layer) \n");
                        //считаем ошибку на последнем слое
                        string s2 = "";
                        for (int j = 0; j < num_of_neurons_in_particular_layer[num_of_layers - 1]; ++j)
                        {
                            ner[num_of_layers - 1][j].err = d_n[j] - ner[num_of_layers - 1][j].y;
         //                   s2 += ner[num_of_layers - 1][j].err.ToString() + "  ,  ";
                        }
         //               Console.WriteLine(s2);
                        //Console.WriteLine("END OF FORWARD");

                        //проверка сходимости
                        List<double> curr_err = new List<double> { };
                        for (int i = 0; i < num_of_neurons_in_particular_layer[num_of_layers - 1]; ++i)
                        {
                            curr_err.Add(ner[num_of_layers - 1][i].err);
                        }
                        bool answer = barrier_function(curr_err, 0.5);
                        // textBox1.AppendText(answer.ToString() + Environment.NewLine);



                        //обратный проход и корректировка весов
                        for (int l = num_of_layers - 1; l > 0; --l)
                        {
                            /*delta на последнем слое*/
                            if (l == num_of_layers - 1)
                            {
                                for (int i = 0; i < num_of_neurons_in_particular_layer[l]; ++i)
                                    ner[l][i].delta = ner[l][i].err * diff_activationFunction(activ_func_num, ner[l][i].e);
                            }
                            else
                            {
                                //delta для всех слоев кроме последнего
                                for (int i = 0; i < num_of_neurons_in_particular_layer[l]; ++i)
                                {
                                    double sum = 0;
                                    for (int j = 0; j < num_of_neurons_in_particular_layer[l + 1]; ++j)
                                        sum += ner[l + 1][j].delta * ner[l + 1][j].w[l];
                                    ner[l][i].delta = diff_activationFunction(activ_func_num, ner[l][i].e) * sum;
                                }
                            }
                            for (int i = 0; i < num_of_neurons_in_particular_layer[l]; ++i)
                            {
                                //корректировка весов
                                for (int j = 0; j < num_of_neurons_in_particular_layer[l - 1]; ++j)
                                {
                                    ner[l][i].w[j] = ner[l][i].w[j] + speed * ner[l][i].delta * ner[l - 1][j].y;
                                    sum_weights_for_epoha +=ner[l][i].w[j];
                                }
                            }
                        }
                        //отдельно обрабатывается первый слой
                        for (int i = 0; i < num_of_neurons_in_particular_layer[0]; ++i)
                            for (int j = 0; j < N; ++j) //на первом слое , поэтому до x_n.Count
                            {
                                ner[0][i].w[j] = ner[0][i].w[j] + speed * ner[0][i].delta * input_images[p].components[j];
                                sum_weights_for_epoha += ner[0][i].w[j];
                            }



                        //чисто для отладки
        /*                Console.WriteLine("\n-> OUTPUT Y (last leyer) \n");
                        string s3 = "";
                        for (int j = 0; j < num_of_neurons_in_particular_layer[num_of_layers - 2]; ++j)
                            s3 += ner[num_of_layers - 2][j].y.ToString() + "  ,  ";
                        Console.WriteLine(s3 + "\n");//*/
                        //чисто для отладки



           //             Console.WriteLine("END OF BACKWARD\n");
                        //проходимся по всем нейронам последнего слоя и находим срднеквадратическую ошибку для изображения
                        double mid_err_image = 0;
                        for (int j = 0; j < num_of_neurons_in_particular_layer[num_of_layers - 1]; ++j)
                            mid_err_image += Math.Pow(ner[num_of_layers - 1][j].err, 2);
                        images_err.Add(mid_err_image);
           //           Console.WriteLine("END OF IMAGE " + p + " PROCESSING");
                    }//по изображениям

                    /*после эпохи считаем суммарную среднеквадратическую ошибку
                    нейронов сети, если она достаточно мала, то заканчиваем обучение*/
                    //пробегаем по всем нейронам
                    double epoha_err = 0;
                    for (int i = 0; i < images_err.Count; ++i)
                    {
                        epoha_err += images_err[i];
                    }
                    epoha_err = epoha_err / (2.0 * size_elems_by_epoch);


                    //перемешиваем образцы
                    input_images = input_images.OrderBy(v => rnd1.Next()).ToList();

                    

                    //еще один критерий останова, если среднеквадратическая ошибка не сильно меняется от эпохи к эпохе
                    //ON
                    /*double epoha_treshhold = 0.2;
                    if (prev_epoha_mid_err_2 - prev_epoha_mid_err_1 < epoha_treshhold && epoha_err - prev_epoha_mid_err_2 < epoha_treshhold)
                        end_criteria = true;
                    prev_epoha_mid_err_1 = prev_epoha_mid_err_2;
                    prev_epoha_mid_err_2 = epoha_err;*/
                    //критерий остановки
                    //сумма весов почти не меняется от эпохи к эпохи
                    double eps_sum_weights = 0.001;
                    if (Math.Abs(sum_weights_for_epoha - prev_sum_weights) < eps_sum_weights)
                        end_criteria = true;
                    prev_sum_weights = sum_weights_for_epoha;
                    sum_weights_for_epoha = 0;

                    //критерий малости mid_err
                    if (epoha_err < stop_epoha_err)
                        end_criteria = true;                   
                    //по максимальной итерации
                    if (epoha_count == stop_num_epoh)
                        end_criteria = true;
                    
                    last_err = epoha_err;
                    Console.WriteLine("Epoha № " + epoha_count + ",_epoha_err = " + epoha_err);
                    ++epoha_count;
                }//по эпохам
                Console.WriteLine("Count of epohas = " + (epoha_count - 1) + ", epoha LAST_err = " + last_err);
                //если сеть обучилась, то сохраняем веса в файл
                try
                {
                    file_with_weights = "NumImages=" + size_elems_by_epoch + ",_epohaCount=" + epoha_count + ",_err=" + last_err + ".txt";
                    using (StreamWriter sw = new StreamWriter(File.Open(file_with_weights, FileMode.Create)))
                    {
                        sw.WriteLine(N); //сколько входных сигналов
                        sw.WriteLine(d_n.Count); // cколько выходных сигналов
                        sw.WriteLine(num_of_hid_layers); //cколько скрытых слоев
                        for (int i = 0; i < num_of_hid_layers; ++i)
                            sw.WriteLine(num_of_neurons_in_particular_layer[i]);

                        for (int i = 0; i < num_of_layers; ++i)//+1
                            for (int j = 0; j < num_of_neurons_in_particular_layer[i]; ++j)
                                for (int k = 0; k < ner[i][j].w.Count; ++k)
                                    sw.Write(ner[i][j].w[k] + " ");
                    }
                }
                catch (Exception ex)
                {}
                Console.Write("If you want train the net input 1, to exit press 0 ->");
                num_process = Convert.ToInt32(Console.ReadLine());
                //num_process = 1;
            }
            
            if (num_process == 1)
            {   //сеть уже обучена, есть файл с весами, достаем веса, создаем сеть и запускаем прямой проход по изображениям для проверки
                int input_ner_1 = 0, output_ner_1 = 0, num_of_hid_layers_1 = 0, num_of_layers_1 = 0;
                int[] check_num_of_neurons_in_particular_layer = new int[] { };
                List<double> weights_from_file = new List<double> { };

               // file_with_weights = "NumImages=16000,_epohaCount=301,_err=0,465929569984108.txt";
                file_with_weights = "50percent,NumImages=1000,_epohaCount=301,_err=0,293245667575368.txt";
                string input_file = path + file_with_weights;
                //Console.WriteLine(input_file);
                //парсим файл
                try
                {
                    using (StreamReader sr = new StreamReader(input_file, Encoding.Default))
                    {
                        input_ner_1 = Convert.ToInt32(sr.ReadLine()); //сколько входных сигналов
                        output_ner_1 = Convert.ToInt32(sr.ReadLine()); // cколько выходных сигналов
                        num_of_hid_layers_1 = Convert.ToInt32(sr.ReadLine());//cколько скрытых слоев
                        num_of_layers_1 = num_of_hid_layers_1 + 1;
                        check_num_of_neurons_in_particular_layer  = new int[num_of_layers_1];
                        for (int i = 0; i < num_of_hid_layers_1; ++i)
                            check_num_of_neurons_in_particular_layer[i] = Convert.ToInt32(sr.ReadLine());
                        check_num_of_neurons_in_particular_layer[num_of_hid_layers_1] = output_ner_1;
                        string all_w = sr.ReadLine();
                       // Console.WriteLine(all_w);
                        string[] weights = all_w.Split(' ');

                        for(int i = 0; i < weights.Length; ++i)
                            weights_from_file.Add(double.Parse(weights[i]));//*/
                    }
                }
                catch (Exception ex)
                {
                    //problem
      //              Console.WriteLine("Wrong destination of file!\n" + input_file + "\n" + ex);
                }
                
                //создаем сеть по данным параметрам 
                List<List<neuron>> check_ner = new List<List<neuron>> { };
                List<double> check_err = new List<double> { };
                
                List<double> check_ner_res = new List<double> { };

                int check_curr_size = input_ner_1;
                int  start_in_w = 0;
                //создание сети
                for (int l = 0; l < num_of_layers_1; ++l) //+1
                {
                    List<neuron> check_ner_l = new List<neuron> { };

                    for (int j = 0; j < check_num_of_neurons_in_particular_layer[l]; ++j)
                    {
                        List<double> weights_to_constructor = new List<double> { };
                        for (int k = 0; k < check_curr_size; ++k)
                            weights_to_constructor.Add(weights_from_file[k + j * check_curr_size + start_in_w]);
                        neuron n = new neuron(weights_to_constructor);
                        check_ner_l.Add(n);
                        weights_to_constructor.Clear();
                    }
                   
                    check_ner.Add(check_ner_l);
                    //prev_layer = check_curr_size;
                    start_in_w += check_curr_size * check_num_of_neurons_in_particular_layer[l];
                    check_curr_size = check_num_of_neurons_in_particular_layer[l];
                }
              /*  for (int i = 0; i < weights_from_file.Count; ++i)
                    Console.WriteLine(weights_from_file[i]);//*/
                
                //для отладки вывести все веса
     /*           for (int l = 0; l < num_of_layers_1; ++l) //+1
                {
                    for (int j = 0; j < check_num_of_neurons_in_particular_layer[l]; ++j)
                    {
                        string check_s = "";
                        for (int k = 0; k < check_ner[l][j].w.Count; ++k)
                            check_s += check_ner[l][j].w[k] + " ";
                        Console.WriteLine(check_s);
                    }
                }
             
                //выше работает!!!
                //запускаем прямой ход

                /*Создание списка изображений букв для проверки*/
                int num_images = 400;
                check_images = process_input_images(16000, num_images, images_location);//считываем с начала файла, поэтому 0
                int check_N = check_images[0].components.Count;
                int check_len = check_N;
                List<double> check_images_err = new List<double> { };
                List<double> check_ner_input_y = new List<double> { };
                int right_img = 0, wrong_img = 0;
                int check_activ_func_num = 1;
                //цикл по изображениям
                for (int p = 0; p < num_images; ++p)
                {
                    List<int> check_d_n = new List<int> { };

                    check_d_n = check_images[p].d_n;
                    //attention здесь где-то ошибка, чек нейрон пустой ходит
                    //прямой ход
                    check_ner_input_y.Clear();
                    for (int j = 0; j < check_N; ++j)
                        check_ner_input_y.Add(check_images[p].components[j] / 15); //  /15 - нормировка

                    for (int l = 0; l < num_of_layers_1; ++l) //+1
                    {
                        for (int j = 0; j < check_num_of_neurons_in_particular_layer[l]; ++j)
                        {
                            check_ner[l][j].neuron_activation(check_ner_input_y, activationFunction, check_activ_func_num);
                            check_ner_res.Add(check_ner[l][j].y);
                        }
                        check_ner_input_y.Clear();
                        for (int i = 0; i < check_ner_res.Count; ++i)
                            check_ner_input_y.Add(check_ner_res[i]);
                        check_ner_res.Clear();
                    }

                    //проверка соответствия y на последнем слое(выхода) и нужного результата
                    double max_y = -100;
                    int max_ind = -1;
                    //ищем максимум на выходах всех нейронов
                    for (int i = 0; i < check_num_of_neurons_in_particular_layer[num_of_hid_layers_1]; ++i)
                    {
                        if (check_ner[num_of_hid_layers_1][i].y > max_y)
                        {
                            max_y = check_ner[num_of_hid_layers_1][i].y;
                            max_ind = i;
                        }
                    }
                    if (check_images[p].d_n[max_ind] == 1)
                        ++right_img;
                    else
                        ++wrong_img;//*/
                }
                Console.WriteLine("The neuron net guess "+ (right_img * 100 / num_images) + "% of provided images(" + num_images + ") !");
               
            }//else if
        }//Main
    }//class program
}
