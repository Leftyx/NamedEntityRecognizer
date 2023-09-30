namespace NamedEntityRecognizer.Models
{
    internal record TokenizationResult
    {
        public string Token { get; init; } = string.Empty;

        public string Label { get; init; } = string.Empty;
    }
}
