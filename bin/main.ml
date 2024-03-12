open Camlgrad;;

let dim1 = 3 in 
let dim2 = 2 in 
let a = Tensor.from_array (Array.make_matrix dim1 dim2 4.0) in
let b = Tensor.create (dim1, dim2) 16.0 in
let c = Tensor.mul a b in

Tensor.backward c;

Tensor.printGrad c;
Tensor.printGrad a;
Tensor.printGrad b;
