toy.func @pattern(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %1 = toy.mul %0, %0 : tensor<3x2xf64>
  toy.return %1 : tensor<3x2xf64>
}