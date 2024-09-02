using System.Diagnostics.CodeAnalysis;

namespace NamedEntityRecognizer.Extensions;

internal static class EnumerableExtension
{
   public static IEnumerable<(T Item, float Probability)> SoftMax<T>(this IEnumerable<T> values, Func<T, float> scoreSelector)
   {
      if (values?.Count() <= 0)
      {
         return [];
      }

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

   public static List<int> FindAllIndexOf<T>(this IEnumerable<T> array, Predicate<T> filter)
   {
      return array.Select((elementInArray, index) => filter(elementInArray) ? index : -1)
                  .Where(index => index != -1)
                  .ToList();
   }
}
