using System.Text.Json.Serialization;

namespace NamedEntityRecognizer.Models
{
    internal sealed class ModelConfiguration
    {
        [JsonPropertyName("_name_or_path")]
        public string NameOrPath { get; set; } = string.Empty;

        [JsonPropertyName("activation")]
        public string? Activation { get; set; }

        [JsonPropertyName("architectures")]
        public List<string>? Architectures { get; set; }

        [JsonPropertyName("attention_dropout")]
        public float? AttentionDropout { get; set; }

        [JsonPropertyName("dim")]
        public int? Dimension { get; set; }

        [JsonPropertyName("dropout")]
        public float? Dropout { get; set; }

        [JsonPropertyName("hidden_dim")]
        public int? HiddenDimension { get; set; }

        [JsonPropertyName("id2label")]
        public Dictionary<string, string> IdTolabel { get; set; } = new();

        [JsonPropertyName("initializer_range")]
        public float InitializerRange { get; set; } = 0.0f;

        [JsonPropertyName("label2id")]
        public Dictionary<string, int> LabelToid { get; set; } = new();

        [JsonPropertyName("max_position_embeddings")]
        public int MaxPositionEmbeddings { get; set; } = 512;

        [JsonPropertyName("model_type")]
        public string ModelType { get; set; } = string.Empty;

        [JsonPropertyName("n_heads")]
        public int? NHeads { get; set; }

        [JsonPropertyName("n_layers")]
        public int? NLayers { get; set; }

        [JsonPropertyName("output_past")]
        public bool? OutputPast { get; set; }

        [JsonPropertyName("pad_token_id")]
        public int? PadTokenId { get; set; }

        [JsonPropertyName("qa_dropout")]
        public float? QaDropout { get; set; } = 0.0f;

        [JsonPropertyName("seq_classif_dropout")]
        public float? SeqClassifDropout { get; set; } = 0.0f;

        [JsonPropertyName("sinusoidal_pos_embds")]
        public bool? SinusoidalPosEmbds { get; set; } = false;

        [JsonPropertyName("tie_weights_")]
        public bool? TieWeights { get; set; } = false;

        [JsonPropertyName("transformers_version")]
        public string TransformersVersion { get; set; } = string.Empty;

        [JsonPropertyName("vocab_size")]
        public int VocabularySize { get; set; } = 0;
    }
}
