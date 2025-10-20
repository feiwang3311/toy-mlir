// Test input: Multiple instances of transpose-mul pattern
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]> : tensor<3x2xf64>

  // First instance of transpose-mul pattern
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>

  // Second instance of transpose-mul pattern
  %4 = toy.transpose(%1 : tensor<3x2xf64>) to tensor<2x3xf64>
  %5 = toy.mul %4, %4 : tensor<2x3xf64>

  // Third instance of transpose-mul pattern
  %6 = toy.transpose(%3 : tensor<3x2xf64>) to tensor<2x3xf64>
  %7 = toy.mul %6, %6 : tensor<2x3xf64>

  toy.print %7 : tensor<2x3xf64>
  toy.return
}
