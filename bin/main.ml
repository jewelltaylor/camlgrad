open Camlgrad;;

let dim1 = 3 in 
let dim2 = 2 in 
let dim3 = 6 in
let a = Tensor.from_array (Array.make_matrix dim1 dim2 4.0) in
let b = Tensor.create (dim2, dim3) 16.0 in
let c = Tensor.matmul a b in
let d = Tensor.sum c in 
Tensor.backward d;

Tensor.printVals c;
Tensor.printVals d;
Tensor.printGrad b;
Tensor.printGrad a;
