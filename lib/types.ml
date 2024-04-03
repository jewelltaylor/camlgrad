open Tensor

type mlp_layer = {
  weights : tensor;
  bias : tensor;
  activation: tensor -> tensor 
}

type mlp = mlp_layer array
