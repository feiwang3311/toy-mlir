// Pattern: transpose result used in both add and mul (DAG structure)
toy.func @pattern(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %1 = toy.add %0, %arg1 : tensor<3x2xf64>
  %2 = toy.mul %0, %1 : tensor<3x2xf64>
  toy.return %2 : tensor<3x2xf64>
}
