// Negative test 1: transpose-mul pattern but with DIFFERENT operands
// Pattern expects: transpose(%x) then mul(transpose_result, transpose_result)
// This has: transpose(%x) then mul(transpose_result, different_value)
// Should NOT match transpose_mul pattern
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]> : tensor<3x2xf64>

  // This should NOT match: mul uses %2 and %1 (different values)
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %1 : tensor<3x2xf64>

  toy.print %3 : tensor<3x2xf64>
  toy.return
}
