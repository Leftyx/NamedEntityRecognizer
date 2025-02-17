using System.Text;
using NamedEntityRecognizer.Models;

namespace NamedEntityRecognizer;

internal partial class Program
{
   static async Task Main()
   {
      CancellationTokenSource cancellationToken = new();

      var sentence = @"My name is Wolfgang Müller and I live in Berlin, Germany.";

      await RunModelWithoutTokenTypeIdsAsync(sentence, cancellationToken);

      await RulModelWithTokenTypeIdsAsync(sentence, cancellationToken);

      Console.WriteLine("");
      Console.WriteLine("Finished!");

      Console.ReadLine();
   }

   private static async Task RunModelWithoutTokenTypeIdsAsync(string sentence, CancellationTokenSource cancellationToken)
   {
      var modelPath = @"<root_folder>\distilbert-base-multilingual-cased-mapa_coarse-ner";

      var configuration = new Configuration(modelPath, numberOfTokens: 5)
      {
         HasTokenTypeIds = false
      };

      var result = await new NerProcessor(configuration)
                          .ProcessAsync(sentence, cancellationToken.Token);

      Console.WriteLine(new string('-', 80));

      result?.ForEach(p =>
      {
         Console.WriteLine("{0}={1}", p.Token, p.Label);
      });

      var listOfTokens = ExtractTokens(result?.Select(p => new Classification(p.Token, p.Label))?.ToList() ?? [], "PERSON");

      foreach (var token in listOfTokens)
      {
         Console.WriteLine(">>> Token (PERSON): {0}", token);
      }

      Console.WriteLine(new string('-', 80));
   }

   private static async Task RulModelWithTokenTypeIdsAsync(string sentence, CancellationTokenSource cancellationToken)
   {
      var modelPath = @"<root_folder>\wikineural-multilingual-ner";

      var configuration = new Configuration(modelPath, numberOfTokens: 5)
      {
         HasTokenTypeIds = true
      };

      var result = await new NerProcessor(configuration)
                          .ProcessAsync(sentence, cancellationToken.Token);

      Console.WriteLine(new string('-', 80));

      result?.ForEach(p =>
      {
         Console.WriteLine("{0}={1}", p.Token, p.Label);
      });

      var listOfTokens = ExtractTokens(result?.Select(p => new Classification(p.Token, p.Label))?.ToList() ?? [], "PER");

      foreach (var token in listOfTokens)
      {
         Console.WriteLine(">>> Token (PERSON): {0}", token);
      }

      Console.WriteLine(new string('-', 80));
   }

   private static List<string> ExtractTokens(IEnumerable<Classification> nerTags, string label, string beginningOfEntityPrefix = "B-", string containedInsideSameEntityPrefix = "I-")
   {
      var names = new List<string>();
      var currentName = new StringBuilder();

      foreach (var nerItem in nerTags)
      {
         if (nerItem.Label.StartsWith(beginningOfEntityPrefix + label))
         {
            if (currentName.Length > 0)
            {
               names.Add(currentName.ToString().Trim());
               currentName.Clear();
            }
            currentName.Append(nerItem.Token);
         }
         else if (nerItem.Label.StartsWith(containedInsideSameEntityPrefix + label) && currentName.Length > 0)
         {
            currentName.Append(nerItem.Token);
         }
      }

      if (currentName.Length > 0)
      {
         names.Add(currentName.ToString().Trim());
      }

      return names;
   }
}


