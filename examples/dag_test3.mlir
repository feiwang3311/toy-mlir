// Test input: add result used in multiple muls
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf64>
  %1 = toy.constant dense<[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf64>
  %2 = toy.constant dense<[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]> : tensor<3x2xf64>

  // This should match the add_mul_chain pattern
  %3 = toy.add %0, %1 : tensor<3x2xf64>
  %4 = toy.mul %3, %2 : tensor<3x2xf64>
  %5 = toy.mul %3, %4 : tensor<3x2xf64>

  toy.print %5 : tensor<3x2xf64>
  toy.return
}
