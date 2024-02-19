open Types


let printVals (tnsr: tensor) = Values.print tnsr.vals 
let printGrad (tnsr: tensor) = Values.print tnsr.grad
