open Camlgrad;;

let dim1 = 3 in 
let dim2 = 2 in 
let a = Values.from_array (A2D (Array.make_matrix dim1 dim2 3.0)) in
let b = Values.from_array (A2D (Array.make_matrix dim1 dim2 3.0)) in

let c = Values.mul a b in 
let d = Values.sum c in
Printf.printf "%f" d;;

