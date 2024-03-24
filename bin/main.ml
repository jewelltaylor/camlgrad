let a = Tensor.random (6, 4) in
let b = Tensor.random (6, 4) in
let c = Tensor.add a b in

let r = Tensor.sum c in
Tensor.backward r;

Tensor.printVals c;
Tensor.printGrad a; 
Tensor.printGrad r; 
