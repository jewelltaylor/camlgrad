open Camlgrad;;

let d = Tensor.from_array (Tensor.A2D (Array.make_matrix 2 2 2.0)) in
let e = Tensor.from_array (Tensor.A2D (Array.make_matrix 2 2 3.0)) in

let f = Tensor.dotT2D d e in 
Printf.printf "%f" f;;
