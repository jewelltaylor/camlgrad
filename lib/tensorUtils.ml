open Types

let get_grad curr_grad dims =
  match curr_grad with
  | GRAD grad -> grad
  | NONE -> Values.create dims 0.0

let get_r_grad tnsr =
  match tnsr.grad with
  | GRAD grad -> grad
  | _ ->  raise TypeException

let printVals (tnsr: tensor) =
  let adim1, adim2 = Values.dim tnsr.vals in
  Values.print tnsr.vals;
  Printf.printf "<values: id=%d shape=(%d, %d)> \n" tnsr.tid adim1 adim2

let printGrad (tnsr: tensor) = 
  match tnsr.grad with
  | GRAD grad -> begin 
    let adim1, adim2 = Values.dim grad in
    Values.print grad;
  Printf.printf "<gradient: id=%d shape=(%d, %d)> \n" tnsr.tid adim1 adim2
  end
  | NONE -> print_endline "NONE"
