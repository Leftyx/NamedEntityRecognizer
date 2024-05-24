using NamedEntityRecognizer.Models;

namespace NamedEntityRecognizer
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            CancellationTokenSource cancellationToken = new();

            var modelPath = @"<root_folder>\distilbert-base-multilingual-cased-mapa_coarse-ner";

            var sentence = @"My name is Wolfgang Müller and I live in Berlin, Germany.";

            var configuration = new Configuration(modelPath, numberOfTokens: 5)
            {
                HasTokenTypeIds = false
            };

            var result = await new NerProcessor(configuration)
                                .ProcessAsync(sentence, cancellationToken.Token);

            result?.ForEach(p =>
            {
                Console.WriteLine("{0}={1}", p.Token, p.Label);
            });

            Console.WriteLine("");
            Console.WriteLine("Finished!");
            Console.ReadLine();
        }
    }
}
