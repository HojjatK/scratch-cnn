using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ScratchCNN
{
    public static class NeuralNetwork_ImageClassifier
    {
        const int ImageWidthHeight = 28;
        static Sample[] trainingData = null, testingData = null;

        public static void Run()
        {
            trainingData = ImageSample.LoadTrainingImages();   // 50,000 training images
            testingData = ImageSample.LoadTestingImages();     // 10,000 testing images

            var net = new NeuralNet(ImageWidthHeight * ImageWidthHeight, 20, 10);
            var trainer = new Trainer(net);

            var sw = new Stopwatch();
            sw.Start();
            trainer.Train(trainingData, testingData, learningRate: .01, epochs: 10);
            sw.Stop();
            Console.WriteLine($"Training Ellapsed: {sw.Elapsed.TotalSeconds} seconds = {sw.Elapsed.TotalSeconds / 60} minutes");

            var testInfos = GetImageTestInfo(new FiringNet(net), testingData).ToList();
            var failures =
                (from testInfo in testInfos
                 where !testInfo.IsCorrect
                 select new { testInfo.ImageSample.Label, testInfo.TotalLoss, testInfo.OutputValues }).ToList();

            Console.WriteLine($"Test set accuracy: { (testInfos.Count - failures.Count) * 100 / testInfos.Count}");

            Console.WriteLine("Failures with highest loss");
            foreach (var f in failures.OrderByDescending(f => f.TotalLoss).Take(100))
            {
                Console.WriteLine($"Label: {f.Label}, Predicted: {HelperMethods.IndexOfMax(f.OutputValues)}, TotalLoss: {f.TotalLoss}");
            }
        }

        static IEnumerable<TestInfo> GetImageTestInfo(FiringNet firingNet, Sample[] samples)
        {
            foreach (ImageSample sample in samples)
            {
                firingNet.FeedForward(sample.Data);
                yield return new TestInfo(sample, firingNet.OutputValues.ToArray());
            }
        }
    }


    class TestInfo
    {
        public readonly ImageSample ImageSample;
        public readonly double[] OutputValues;

        public bool IsCorrect => ImageSample.IsOutputCorrect(OutputValues);

        public double TotalLoss => OutputValues
            .Select((v, i) => (v - (i == ImageSample.Label ? 1 : 0)) * (v - (i == ImageSample.Label ? 1 : 0)) / 2)
            .Sum();

        //Lazy<System.Drawing.Image> _image;
        //public System.Drawing.Image Image => _image.Value;

        public TestInfo(ImageSample imageSample, double[] outputValues)
        {
            ImageSample = imageSample;
            OutputValues = outputValues;
            //_image = new Lazy<System.Drawing.Image>(() => ToImage(ImageSample.Pixels, 0, ImageWidthHeight, ImageWidthHeight));
        }
    }

    #region Neural Network Classes
    class Neuron
    {
        public readonly NeuralNet Net;
        public readonly int Layer, Index;

        public double[] InputWeights;
        public double Bias;

        public Activator Activator => Net.Activators[Layer];

        public bool IsOutputNeuron => Layer == Net.Neurons.Length - 1;

        static readonly Random _random = new Random();

        static double GetSmallRandomNumber() =>
            (.0009 * _random.NextDouble() + .0001) * (_random.Next(2) == 0 ? -1 : 1);

        public Neuron(NeuralNet net, int layer, int index, int inputWeightCount)
        {
            Net = net;
            Layer = layer;
            Index = index;

            Bias = GetSmallRandomNumber();
            InputWeights = Enumerable.Range(0, inputWeightCount).Select(_ => GetSmallRandomNumber()).ToArray();
        }
    }

    class NeuralNet
    {
        public readonly Neuron[][] Neurons;     // Layers of neurons
        public Activator[] Activators;          // Activators for each layer

        public NeuralNet(params int[] neuronsInEachLayer)   // including the input layer
        {
            Neurons = neuronsInEachLayer
                .Skip(1)                          // Skip the input layer
                .Select((count, layer) =>
                   Enumerable.Range(0, count)
                             .Select(index => new Neuron(this, layer, index, neuronsInEachLayer[layer]))
                             .ToArray())
                .ToArray();

            // Default to ReLU activators
            Activators = Enumerable
                .Repeat((Activator)new ReLUActivator(), neuronsInEachLayer.Length - 1)
                .ToArray();
        }
    }

    class FiringNeuron
    {
        public readonly Neuron Neuron;

        public double TotalInput, Output;
        public double InputVotes, OutputVotes;   // Votes for change = slope of the loss vs input/output

        public FiringNeuron(Neuron neuron) => Neuron = neuron;

        public void ComputeTotalInput(double[] inputValues)
        {
            double sum = 0;

            for (int i = 0; i < Neuron.InputWeights.Length; i++)
                sum += inputValues[i] * Neuron.InputWeights[i];

            TotalInput = Neuron.Bias + sum;
        }

        public unsafe void AdjustWeightsAndBias(double[] inputValues, double learningRate)
        {
            double adjustment = InputVotes * learningRate;

            lock (Neuron) Neuron.Bias += adjustment;

            int max = Neuron.InputWeights.Length;

            fixed (double* inputs = inputValues)
            fixed (double* weights = Neuron.InputWeights)
            lock (Neuron.InputWeights)
                for (int i = 0; i < max; i++)
                    //Neuron.InputWeights[i] += adjustment * inputValues[i];
                    //Using pointers avoids bounds-checking and so reduces the time spent holding the lock.
                    *(weights + i) += adjustment * *(inputs + i);
        }
    }
    #endregion

    #region Learning

    class FiringNet
    {
        public readonly NeuralNet Net;
        public FiringNeuron[][] Neurons;
        FiringNeuron[][] NeuronsWithLayersReversed;

        public FiringNeuron[] OutputLayer => Neurons[Neurons.Length - 1];

        public IEnumerable<double> OutputValues => OutputLayer.Select(n => n.Output);

        public FiringNet(NeuralNet net)
        {
            Net = net;
            Neurons = Net.Neurons.Select(layer => layer.Select(n => new FiringNeuron(n)).ToArray()).ToArray();
            NeuronsWithLayersReversed = Neurons.Reverse().ToArray();
        }

        public void FeedForward(double[] inputValues)
        {
            int i = 0;
            foreach (var layer in Neurons)
            {
                foreach (var neuron in layer)
                    neuron.ComputeTotalInput(inputValues);

                Net.Activators[i++].ComputeOutputs(layer);

                // The outputs for this layer become the inputs for the next layer.
                if (layer != OutputLayer)
                    inputValues = layer.Select(l => l.Output).ToArray();
            }
        }

        public void Learn(double[] inputValues, double[] desiredOutputs, double learningRate)
        {
            FeedForward(inputValues);

            FiringNeuron[] layerJustProcessed = null;

            // Calculate all the output and input votes.
            foreach (var layer in NeuronsWithLayersReversed)
            {
                bool isOutputLayer = layerJustProcessed == null;
                foreach (var neuron in layer)
                {
                    if (isOutputLayer)
                        // For neurons in the output layer, the loss vs output slope = -error.
                        neuron.OutputVotes = desiredOutputs[neuron.Neuron.Index] - neuron.Output;
                    else
                        // For hidden neurons, the loss vs output slope = weighted sum of next layer's input slopes.
                        neuron.OutputVotes =
                            layerJustProcessed.Sum(n => n.InputVotes * n.Neuron.InputWeights[neuron.Neuron.Index]);

                    // The loss vs input slope = loss vs output slope times activation function slope (chain rule).
                    neuron.InputVotes = neuron.OutputVotes * neuron.Neuron.Activator.GetActivationSlopeAt(neuron);
                }
                layerJustProcessed = layer;
            }

            // We can improve the training by scaling the learning rate by the layer index.
            int learningRateMultiplier = Neurons.Length;

            // Updates weights and biases.
            foreach (var layer in Neurons)
            {
                foreach (var neuron in layer)
                    neuron.AdjustWeightsAndBias(inputValues, learningRate * learningRateMultiplier);

                if (layer != OutputLayer)
                    inputValues = layer.Select(l => l.Output).ToArray();

                learningRateMultiplier--;
            }
        }
    }

    class Trainer
    {
        Random _random = new Random();

        public readonly NeuralNet Net;
        public int CurrentEpoch;
        public double CurrentAccuracy;
        public int Iterations;
        public string TrainingInfo;

        public Trainer(NeuralNet net) => Net = net;

        public void Train(Sample[] trainingData, Sample[] testingData, double learningRate, int epochs)
        {
            _random = new Random();
            var trainingSet = trainingData.ToArray();

            TrainingInfo = $"Learning rate = {learningRate}";

            for (CurrentEpoch = 0; CurrentEpoch < epochs; CurrentEpoch++)
            {
                Console.Write($"Training epoch {CurrentEpoch}... ");
                CurrentAccuracy = TrainEpoch(trainingSet, learningRate);
                learningRate *= .9;   // This help to avoids oscillation as our accuracy improves.
                Console.WriteLine("Done. Training accuracy = " + CurrentAccuracy.ToString("N1") + "%");
            }

            string testAccuracy = ((Test(new FiringNet(Net), testingData) * 100).ToString("N1") + "%");
            TrainingInfo += $"\r\nTotal epochs = {CurrentEpoch}\r\nFinal test accuracy = {testAccuracy}";
        }

        public double TrainEpoch(Sample[] trainingData, double learningRate)
        {
            Shuffle(_random, trainingData);   // For each training epoch, randomize order of the training samples.

            // One FiringNet per thread to avoid thread-safety problems.
            var trainer = new ThreadLocal<FiringNet>(() => new FiringNet(Net));
            Parallel.ForEach(trainingData, CancellableParallel, sample =>
            {
                trainer.Value.Learn(sample.Data, sample.ExpectedOutput, learningRate);
                Interlocked.Increment(ref Iterations);
            });

            return Test(new FiringNet(Net), trainingData.Take(10000).ToArray()) * 100;
        }

        public double Test(FiringNet firingNet, Sample[] samples)
        {
            int bad = 0, good = 0;
            foreach (var sample in samples)
            {
                firingNet.FeedForward(sample.Data);
                if (sample.IsOutputCorrect(firingNet.OutputValues.ToArray()))
                    good++;
                else
                    bad++;
            }
            return (double)good / (good + bad);
        }

        static void Shuffle<T>(Random random, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = random.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        // We want to cancel any outstanding training when the user cancels or re-runs the query.
        CancellationTokenSource _cancelSource = new CancellationTokenSource();
        ParallelOptions CancellableParallel => new ParallelOptions { CancellationToken = _cancelSource.Token };
        //Trainer() => Util.Cleanup += (sender, args) => _cancelSource.Cancel();

        //object ToDump() => NeuralNetRenderer(this);
    }

    #endregion

    #region Activation
    abstract class Activator
    {
        public abstract void ComputeOutputs(FiringNeuron[] layer);
        public abstract double GetActivationSlopeAt(FiringNeuron neuron);
    }

    class ReLUActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = neuron.TotalInput > 0 ? neuron.TotalInput : neuron.TotalInput / 100;
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron) => neuron.TotalInput > 0 ? 1 : .01;
    }

    class LogisticSigmoidActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = 1 / (1 + Math.Exp(-neuron.TotalInput));
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
            => neuron.Output * (1 - neuron.Output);
    }

    class HyperTanActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = Math.Tanh(neuron.TotalInput);
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
        {
            var tanh = neuron.Output;
            return 1 - tanh * tanh;
        }
    }

    class SoftMaxActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            double sum = 0;

            foreach (var neuron in layer)
            {
                neuron.Output = Math.Exp(neuron.TotalInput);
                sum += neuron.Output;
            }

            foreach (var neuron in layer)
            {
                var oldOutput = neuron.Output;
                neuron.Output = neuron.Output / (sum == 0 ? 1 : sum);
            }
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
        {
            double y = neuron.Output;
            return y * (1 - y);
        }
    }

    class SoftMaxActivatorWithCrossEntropyLoss : SoftMaxActivator  // Use this only on the output layer!
    {
        // This is the derivative after modifying the loss function.
        public override double GetActivationSlopeAt(FiringNeuron neuron) => 1;
    }
    #endregion

    #region Sample data

    class Sample
    {
        public double[] Data;
        public double[] ExpectedOutput;
        public Func<double[], bool> IsOutputCorrect;
    }

    class ImageSample : Sample
    {
        const int categoryCount = 10;

        public byte Label;
        public byte[] Pixels;

        public ImageSample(byte label, byte[] pixels, int categoryCount)
        {
            Label = label;
            Pixels = pixels;
            Data = ToDouble(pixels);
            ExpectedOutput = LabelToDoubleArray(label, categoryCount);
            IsOutputCorrect = input => HelperMethods.IndexOfMax(input) == Label;
        }

        static double[] ToDouble(byte[] data) => data.Select(p => (double)p / 255).ToArray();

        static double[] LabelToDoubleArray(byte label, int categoryCount) =>
            Enumerable.Range(0, categoryCount).Select(i => i == label ? 1d : 0).ToArray();

        public static ImageSample[] LoadTrainingImages() =>
            Load(GetDataFilePath("Training Images", trainingImagesUri), GetDataFilePath("Training Labels", trainingLabelsUri), categoryCount);

        public static ImageSample[] LoadTestingImages() =>
            Load(GetDataFilePath("Testing Images", testingImagesUri), GetDataFilePath("Testing Labels", testingLabelsUri), categoryCount);

        public static ImageSample[] Load(string imgPath, string labelPath, int categoryCount)
        {
            Console.WriteLine($"Loading {System.IO.Path.GetFileName(imgPath)}...");
            var imgData = File.ReadAllBytes(imgPath);
            var header = imgData.Take(16).Reverse().ToArray();
            int imgCount = BitConverter.ToInt32(header, 8);
            int rows = BitConverter.ToInt32(header, 4);
            int cols = BitConverter.ToInt32(header, 0);

            return File.ReadAllBytes(labelPath)
                .Skip(8)  // skip header
                .Select((label, i) => new ImageSample(label, SliceArray(imgData, rows * cols * i + header.Length, rows * cols), categoryCount))
                .ToArray();
        }

        static byte[] SliceArray(byte[] source, int offset, int length)
        {
            var target = new byte[length];
            Array.Copy(source, offset, target, 0, length);
            return target;
        }

        static readonly string basePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LINQPad Machine Learning", "MNIST digits");

        static string SavedDataPath => Path.Combine(basePath, "saved.bin");

        const string
            trainingImagesUri = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            trainingLabelsUri = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            testingImagesUri = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            testingLabelsUri = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

        static string GetDataFilePath(string filename, string uri)
        {
            if (!Directory.Exists(basePath)) Directory.CreateDirectory(basePath);
            string fullPath = Path.Combine(basePath, filename);

            if (!File.Exists(fullPath))
            {
                Console.Write($"Downloading {filename}... ");

                var buffer = new byte[0x10000];
                using (var ms = new MemoryStream(new WebClient().DownloadData(uri)))
                using (var inStream = new GZipStream(ms, CompressionMode.Decompress))
                using (var outStream = File.Create(fullPath))
                    while (true)
                    {
                        int len = inStream.Read(buffer, 0, buffer.Length);
                        if (len == 0) break;
                        outStream.Write(buffer, 0, len);
                    }

                Console.WriteLine("Done");
            }
            return fullPath;
        }
    }
    #endregion

    #region Helper Methods
    static class HelperMethods
    {
        public static int IndexOfMax(double[] values)
        {
            double max = 0;
            int indexOfMax = 0;
            for (int i = 0; i < values.Length; i++)
                if (values[i] > max)
                {
                    max = values[i];
                    indexOfMax = i;
                }
            return indexOfMax;
        }
    }
    #endregion
}
