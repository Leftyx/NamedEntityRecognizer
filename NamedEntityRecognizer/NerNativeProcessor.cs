using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;

namespace NamedEntityRecognizer
{
    internal class NerNativeProcessor
    {
        private class TestSingleSentenceData
        {
            public required string Sentence;
            public required string[] Label;
        }

        private class Label
        {
            public string Key { get; set; }
        }

        public void TestSimpleNer()
        {
            MLContext ML = new();

            var labels = ML.Data.LoadFromEnumerable(
                new[] {
                new Label { Key = "PERSON" },
                new Label { Key = "CITY" },
                new Label { Key = "COUNTRY"  }
                });

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new()
                    {   // Testing longer than 512 words.
                        Sentence = "Barbara and Frank live in the Liechtenstein",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                    },
                     new()
                     {
                        Sentence = "Alice and Bob live in the USA",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                     },
                }));

            var chain = new EstimatorChain<ITransformer>();

            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
               .Append(ML.MulticlassClassification.Trainers.NameEntityRecognition(outputColumnName: "outputColumn"))
               .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            // var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            var transformer = estimator.Fit(dataView);

            // var transformerSchema = transformer.GetOutputSchema(dataView.Schema);
            // var predictionIDV = transformer.Transform(test);

            transformer.Dispose();
        }
    }
}
