using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ScratchCNN
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork_ImageClassifier.Run();
            Console.WriteLine("Press any key to exit...");
            Console.ReadLine();
        }
    }
}
