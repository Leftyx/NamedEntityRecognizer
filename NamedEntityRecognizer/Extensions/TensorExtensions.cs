using Microsoft.ML.OnnxRuntime.Tensors;

namespace NamedEntityRecognizer.Extensions;

internal static class TensorExtensions
{
   [Obsolete]
   public static Tensor<long> ConvertToTensor(long[] inputArray, int dimension)
   {
      Tensor<long> input = new DenseTensor<long>([1, dimension]);

      for (var i = 0; i < inputArray.Length; i++)
      {
         input[0, i] = inputArray[i];
      }

      return input;
   }
}
