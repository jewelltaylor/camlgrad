open Camlgrad;;

let dim1 = 3 in 
let dim2 = 2 in 
let dim3 = 6 in
let a = Tensor.from_array (Array.make_matrix dim1 dim2 4.0) in
let b = Tensor.create (dim2, dim3) 16.0 in
let c = Tensor.matmul a b in
print_endline "hello1";
c.grad <- GRAD (Values.create (dim1, dim3) 1.0); 
print_endline "hello2";
Tensor.backward c;

Tensor.printGrad c;
Tensor.printGrad a;
Tensor.printGrad b;
