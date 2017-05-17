using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace task3_console
{
    public class LetterImage
    {
        public List<double> components = new List<double> { };
        public List<int> d_n = new List<int> { };
        public char label;

        public LetterImage(string s)
        {
            this.label = s[0];
            s = s.Substring(2);
            string[] new_s = s.Split(',');
            for (int i = 0; i < new_s.Length; ++i)
                this.components.Add(int.Parse(new_s[i]));
            int position = this.label - 65;
            for (int i = 0; i < 26; ++i)
                d_n.Add(0);
            d_n[position] = 1;
        }
    }
}
