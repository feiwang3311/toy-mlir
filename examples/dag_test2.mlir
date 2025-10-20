// Test input: Two transposes multiplied together
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]> : tensor<2x3xf64>

  // This should match the double_transpose pattern
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>

  toy.print %4 : tensor<3x2xf64>
  toy.return
}
