open Tensor

exception InvalidArgumentException of string

type mlp_layer = {
  weights : tensor;
  bias : tensor;
  activation: tensor -> tensor 
}

type mlp = mlp_layer array
