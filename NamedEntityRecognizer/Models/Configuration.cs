namespace NamedEntityRecognizer.Models
{
    internal sealed class Configuration
    {
        public Configuration(string modelPath)
        {
            ModelPath = modelPath;
        }

        public Configuration(string modelPath, int numberOfTokens)
        {
            ModelPath = modelPath;
            NumberOfTokens = numberOfTokens;
        }

        /// <summary>
        /// Path for the Bert model
        /// </summary>
        public string ModelPath { get; init; } = string.Empty;

        /// <summary>
        /// Maximum N. of Tokens
        /// </summary>
        public int MaximumNumberOfTokens { get; set; } = 512;

        /// <summary>
        /// Indicates if the model contains token_type_ids
        /// </summary>
        public bool HasTokenTypeIds { get; set; } = false;

        // Is the BERT model cased ?
        public bool IsCasedModel { get; set; } = true;

        /// <summary>
        /// Number of tokens in the model
        /// For Bert models we should have (tokenizer_config.json)
        ///     "cls_token": "[CLS]",
        ///     "mask_token": "[MASK]",
        ///     "pad_token": "[PAD]",
        ///     "sep_token": "[SEP]",
        ///     "unk_token": "[UNK]"
        /// </summary>
        public int NumberOfTokens { get; set; } = 5;
    }
}
