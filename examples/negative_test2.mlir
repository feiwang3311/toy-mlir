// Negative test 2: Partial pattern - only transpose, no mul
// Pattern expects: transpose followed by mul
// This has: only transpose operation
// Should NOT match transpose_mul pattern
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>

  // This should NOT match: transpose exists but no mul follows
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>

  toy.print %1 : tensor<3x2xf64>
  toy.return
}
