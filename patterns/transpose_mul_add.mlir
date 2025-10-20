// Pattern: composition of transpose_mul + add
// This pattern matches when a call to transpose_mul (any _N suffix) is followed by add
toy.func @pattern(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = toy.generic_call @transpose_mul(%arg0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  %1 = toy.add %0, %arg1 : tensor<3x2xf64>
  toy.return %1 : tensor<3x2xf64>
}
