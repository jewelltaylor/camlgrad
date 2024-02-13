open Camlgrad;;

let dim1 = 3 in 
let dim2 = 2 in 
let a = Tensor.from_array (Tensor.A2D (Array.make_matrix dim1 dim2 3.0)) in

let b = Tensor.exp a in 
Utils.print2D b;;


