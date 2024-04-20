(* Tensor Interface *)

(* Imports *)
open Values

(* Type Definitions *)

type gradient = 
  | GRAD of values
  | NONE

type tensor = {
  tid : int;
  mutable vals : values;
  mutable grad : gradient;
  mutable acc_grad : gradient;
  op : operator;
} and operator = 
  | ADD of tensor * tensor 
  | SUB of tensor * tensor 
  | MUL of tensor * tensor 
  | DIV of tensor * tensor
  | MATMUL of tensor * tensor
  | NEG of tensor 
  | EXP of tensor 
  | LOG of tensor 
  | SQRT of tensor
  | POW2 of tensor 
  | SUM of tensor 
  | RELU of tensor
  | SIGMOID of tensor
  | CREATE

(* Creating *)
val create : dimensions -> float -> tensor 
val from_array : standard_array -> tensor 
val ones : dimensions -> tensor 
val zeros : dimensions -> tensor 
val random : dimensions -> tensor

(* Utilities *)
val printVals : tensor -> unit
val printGrad : tensor -> unit
val dim : tensor -> dimensions
val get_grad : tensor -> values 
val get_acc_grad : tensor -> values 

(* Unary Operations *)
val neg : tensor -> tensor
val exp : tensor -> tensor
val log : tensor -> tensor
val sqrt : tensor -> tensor
val pow2 : tensor -> tensor
val sum : tensor -> tensor
val relu : tensor -> tensor
val sigmoid : tensor -> tensor

(* Binary Operations *)
val add : tensor -> tensor -> tensor
val sub : tensor -> tensor -> tensor
val mul : tensor -> tensor -> tensor
val div : tensor -> tensor -> tensor
val matmul : tensor -> tensor -> tensor


(* Graph Operations *)
val reverse_topological_sort : tensor -> (tensor -> unit) -> unit
val visualize_computation_graph : ?file_name:string -> tensor -> unit
val backward : tensor -> unit
