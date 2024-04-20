(* Values Interface *) 

open Bigarray

(* Type Definitions *)
type standard_array = float array array
type values = (float, float32_elt, c_layout) Array2.t
type dimensions = int * int

(* Creating *)
val from_array : standard_array -> values
val create : dimensions -> float -> values
val ones : dimensions -> values
val zeros : dimensions -> values
val random : dimensions -> values 

(* Utilities *)
val dim : values -> int * int
val  print : values -> unit


(* Unary Operations *)
val log : values -> values
val exp : values -> values
val sqrt : values -> values
val reciprocal: values -> values
val abs : values -> values
val pow2 : values -> values
val sum : values -> values
val neg : values -> values

(* Binary Operations *)
val dot : values -> values -> values
val div : values -> values -> values
val mul : values -> values -> values
val add : values -> values -> values
val sub : values -> values -> values
val matmul : ?trans_a:int -> ?trans_b:int -> values -> values -> values
