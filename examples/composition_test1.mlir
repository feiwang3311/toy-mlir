// Test input for composition pattern matching
// This simulates IR after transpose_mul pattern has been extracted
// We manually create this to test if transpose_mul_add pattern can match

module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf64>

    // This should match transpose_mul_add pattern:
    // Call to transpose_mul_0 followed by add
    %2 = toy.generic_call @transpose_mul_0(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
    %3 = toy.add %2, %1 : tensor<3x2xf64>

    toy.print %3 : tensor<3x2xf64>
    toy.return
  }

  // The extracted transpose_mul function (from previous pass)
  toy.func private @transpose_mul_0(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %1 = toy.mul %0, %0 : tensor<3x2xf64>
    toy.return %1 : tensor<3x2xf64>
  }
}
