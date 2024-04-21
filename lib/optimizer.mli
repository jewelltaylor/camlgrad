(* Optimizer interface *)

(* Imports *)
open Mlp
open Tensor

(* Optimizer Backward Functions *)
val apply_update : mlp_layer -> float -> unit
val update : mlp -> float -> unit
val gradient_descent : mlp -> tensor -> float -> unit 

(* Optimizer Zero Grad Functions *)
val zero_grad_mlp_layer : mlp_layer -> unit
val zero_grad_mlp : mlp -> unit


