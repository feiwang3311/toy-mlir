// Negative test 3: Wrong operation types
// Pattern expects: transpose then mul
// This has: transpose then ADD (not mul)
// Should NOT match transpose_mul pattern
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>

  // This should NOT match: has transpose and add, but pattern wants transpose and mul
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.add %1, %1 : tensor<3x2xf64>

  toy.print %2 : tensor<3x2xf64>
  toy.return
}
