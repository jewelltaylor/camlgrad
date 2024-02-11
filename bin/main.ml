open Camlgrad;;

let d = Tensor.from_array (Tensor.A2D (Array.make_matrix 2 2 2.0)) in
let e = Tensor.from_array (Tensor.A2D (Array.make_matrix 2 2 3.0)) in

let f = Tensor.addT2D d e in 

match f with 
| (Tensor.T2D t2D) -> List.iter (fun i -> Printf.printf "%f" t2D.{i, 0} ) [0; 1]
| _ -> print_endline "error"
