using System.Text.Json;

namespace NamedEntityRecognizer;

internal abstract class BaseProcessor
{
   internal static readonly JsonSerializerOptions _serializatioOptions = new()
   {
      PropertyNameCaseInsensitive = true
   };

   internal static async Task<T?> LoadConfigurationAsync<T>(string path, CancellationToken cancellationToken = default)
   {
      if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
      {
         return default;
      }

      using FileStream stream = File.OpenRead(path);
      {
         return await JsonSerializer.DeserializeAsync<T>(stream, _serializatioOptions, cancellationToken);
      }
   }
}
