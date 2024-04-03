open Camlgrad.Types

let get_grad_and_init_to_zero_if_none curr_grad dims =
  match curr_grad with
  | GRAD grad -> grad
  | NONE -> Values.create dims 0.0

let get_grad tnsr =
  match tnsr.grad with
  | GRAD grad -> grad
  | _ ->  raise TypeException

let get_acc_grad tnsr =
  match tnsr.acc_grad with
  | GRAD grad -> grad
  | _ ->  raise TypeException

let printVals (tnsr: tensor) =
  let adim1, adim2 = Values.dim tnsr.vals in
  Values.print tnsr.vals;
  Printf.printf "<values: id=%d shape=(%d, %d)> \n" tnsr.tid adim1 adim2

let printGrad (tnsr: tensor) = 
  match tnsr.acc_grad with
  | GRAD grad -> begin 
    let adim1, adim2 = Values.dim grad in
    Values.print grad;
  Printf.printf "<gradient: id=%d shape=(%d, %d)> \n" tnsr.tid adim1 adim2
  end
  | NONE -> print_endline "NONE"

let op_constructor_to_string op = 
  match op with
  | ADD (_, _) -> "ADD"
  | SUB (_, _) -> "SUB"
  | MUL (_, _) -> "MUL"
  | DIV (_, _) -> "DIV"
  | NEG _ -> "NEG"
  | POW2 _-> "POW2"
  | EXP _ -> "EXP"
  | LOG _ -> "LOG"
  | SQRT _ -> "SQRT"
  | SUM _ -> "SUM"
  | MATMUL (_, _) -> "MATMUL"
  | RELU _ -> "RELU"
  | SIGMOID _ -> "SIGMOID"
  | CREATE -> "CREATE"

let tensor_to_vertex_string a = Printf.sprintf "%s\n" (string_of_int a.tid) 
let create_op_to_vertex_string op = Printf.sprintf "%s\n" (op_constructor_to_string op)
let urnary_op_to_vertex_string op a_id = Printf.sprintf "%s %s\n" (op_constructor_to_string op) (string_of_int a_id)
let binary_op_to_vertex_string op a_id b_id = 
  Printf.sprintf "%s %s %s\n" (op_constructor_to_string op) (string_of_int a_id) (string_of_int b_id)
