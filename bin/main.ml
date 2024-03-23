let a = Tensor.create (6, 4) 90.0 in
let b = Tensor.create (6, 4) 10.0 in
let c = Tensor.sub a b in

let r = Tensor.sum c in
print_endline "Here";
Tensor.backward r;

Tensor.printGrad a; 
Tensor.printGrad r; 
