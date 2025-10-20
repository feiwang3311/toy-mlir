// Test input: transpose result used in both add and mul
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf64>

  // This should match the transpose_dag pattern
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.add %2, %1 : tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>

  toy.print %4 : tensor<3x2xf64>
  toy.return
}
