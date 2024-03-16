open Types

let get_grad curr_grad dims =
  match curr_grad with
  | GRAD grad -> grad
  | NONE -> Values.create dims 0.0

let get_r_grad tnsr =
  match tnsr.grad with
  | GRAD grad -> grad
  | _ ->  raise TypeException

let printVals (tnsr: tensor) = Values.print tnsr.vals 
let printGrad (tnsr: tensor) = 
  match tnsr.grad with
  | GRAD grad -> Values.print grad
  | NONE -> print_endline "NONE"
