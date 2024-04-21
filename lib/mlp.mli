(* Mlp Interface *)

(* Imports *)
open Tensor

(* Type Definitions *)
type dimensions = int * int
type mlp_layer = {
  weights : tensor;
  bias : tensor;
  activation: tensor -> tensor 
}
type mlp = mlp_layer array

(* Creating *)
val get_mlp : ((tensor -> tensor) * dimensions) array -> mlp
val get_mlp_layer : ?activation:(tensor -> tensor) -> dimensions -> mlp_layer 

(* Forward Pass *)
val mlp_forward : mlp -> tensor -> (tensor * tensor array)
val mlp_layer_forward : mlp_layer -> tensor -> tensor 

