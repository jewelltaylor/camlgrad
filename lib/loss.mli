(* Loss Interface *)

(* Imports *)
open Tensor

(* Loss Functions *)
val mean_squared_error : tensor -> tensor -> tensor
val binary_cross_entropy : tensor -> tensor -> tensor
