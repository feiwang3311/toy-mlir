// Pattern: add result used in two different muls
toy.func @pattern(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>, %arg2: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = toy.add %arg0, %arg1 : tensor<3x2xf64>
  %1 = toy.mul %0, %arg2 : tensor<3x2xf64>
  %2 = toy.mul %0, %1 : tensor<3x2xf64>
  toy.return %2 : tensor<3x2xf64>
}
