using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;
using NamedEntityRecognizer.Extensions;
using NamedEntityRecognizer.Models;

namespace NamedEntityRecognizer;

internal sealed class NerProcessor : BaseProcessor
{
   private const string VOCAB_FILE = "vocab.txt";
   private const string CONFIG_FILE = "config.json";
   private const string MODEL_FILE = "model.onnx";

   private const string INPUT_IDS = "input_ids";
   private const string ATTENTION_MASK = "attention_mask";
   private const string TOKEN_TYPE_IDS = "token_type_ids";
   private const string LABEL_NO_ENTITY = "O";

   private readonly Configuration _configuration;

   public NerProcessor(Configuration configuration)
   {
      _configuration = configuration;
   }

   internal async Task<List<TokenizationResult>?> ProcessAsync(string sentence, CancellationToken cancellationToken = default)
   {
      List<TokenizationResult>? result = [];

      var vocabularyDocument = Path.Combine(_configuration.ModelPath, VOCAB_FILE);
      var configurationDocument = Path.Combine(_configuration.ModelPath, CONFIG_FILE);
      var bertModel = Path.Combine(_configuration.ModelPath, MODEL_FILE);

      if (!File.Exists(vocabularyDocument))
      {
         throw new FileNotFoundException(nameof(vocabularyDocument));
      }

      if (!File.Exists(configurationDocument))
      {
         throw new FileNotFoundException(nameof(configurationDocument));
      }

      if (!File.Exists(bertModel))
      {
         throw new FileNotFoundException(nameof(bertModel));
      }

      var configuration = await LoadConfigurationAsync<ModelConfiguration>(configurationDocument, cancellationToken);

      if (configuration is null || configuration?.IdTolabel?.Count == 0)
      {
         throw new NullReferenceException(nameof(configuration));
      }

      var labelsCount = configuration!.IdTolabel.Count;

      var tokenizer = new BertTokenizer();

      await tokenizer.LoadVocabularyAsync(vocabularyDocument, convertInputToLowercase: !_configuration.IsCasedModel);

      var (inputIds, attentionMask, tokenTypeIds) = tokenizer.Encode(sentence, _configuration.MaximumNumberOfTokens);

      var tensorInputIds = TensorExtensions.ConvertToTensor(inputIds.ToArray(), inputIds.Length);

      var inputs = new List<NamedOnnxValue> {
             NamedOnnxValue.CreateFromTensor(INPUT_IDS, tensorInputIds),
             NamedOnnxValue.CreateFromTensor(ATTENTION_MASK, TensorExtensions.ConvertToTensor(attentionMask.ToArray(), inputIds.Length)),
         };

      if (_configuration.HasTokenTypeIds)
      {
         inputs.Add(NamedOnnxValue.CreateFromTensor(TOKEN_TYPE_IDS, TensorExtensions.ConvertToTensor(tokenTypeIds.ToArray()!, inputIds.Length)));
      }

      using (InferenceSession session = new(bertModel))
      {
         var output = session.Run(inputs);

         if (output?.Count > 0)
         {
            var batchedResult = (output.ToList().FirstOrDefault()?.Value as IEnumerable<float>)
                ?.Chunk(labelsCount)
                ?.Select(y => y.ArgMax());

            if (batchedResult?.Count() == 0)
            {
               return result;
            }

            var predictedLabels = batchedResult?.Select(res => configuration.IdTolabel?[res.ToString()]);

            result = predictedLabels?.Zip(tensorInputIds, (label, value) =>
            {
               return new TokenizationResult
               {
                  Token = tokenizer.Decode([0, value])[_configuration.NumberOfTokens..],
                  Label = label ?? string.Empty,
               };
            })
            ?.Where(x => x.Label != LABEL_NO_ENTITY)
            ?.ToList();
         }
      }

      return result;
   }
}
