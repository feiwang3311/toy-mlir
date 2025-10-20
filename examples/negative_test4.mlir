// Negative test 4: Correct ops but wrong data flow
// Pattern expects: transpose(%arg0) then mul(transpose_result, transpose_result)
// This has: transpose and mul with correct types, but mul doesn't use transpose result
// Should NOT match transpose_mul pattern
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]> : tensor<3x2xf64>

  // This should NOT match: transpose exists, mul exists, but data flow is wrong
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  // %2 is not used in the mul at all!
  %3 = toy.mul %1, %1 : tensor<3x2xf64>

  toy.print %3 : tensor<3x2xf64>
  toy.return
}
