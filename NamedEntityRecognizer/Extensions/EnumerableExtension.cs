using System.Diagnostics.CodeAnalysis;

namespace NamedEntityRecognizer.Extensions
{
    internal static class EnumerableExtension
    {
        public static IEnumerable<(T Item, float Probability)> SoftMax<T>(this IEnumerable<T> values, Func<T, float> scoreSelector)
        {
            if (values?.Count() <= 0) return Enumerable.Empty<(T, float)>();

            var maxScore = values!.Max(scoreSelector);

            var sum = values!.Sum(r => Math.Exp(scoreSelector(r) - maxScore));

            return values!.Select(r => (r, (float)(Math.Exp(scoreSelector(r) - maxScore) / sum)));
        }

        public static int ArgMax([DisallowNull] this IList<float> values)
        {
            if (values?.Count == 0)
            {
                return 0;
            }

            return values!.IndexOf(values!.Max());
        }
    }
}
