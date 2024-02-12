open Camlgrad;;

let dim1 = 4 in 
let dim2 = 6 in
let dim3 = 2 in
let d = Tensor.from_array (Tensor.A2D (Array.make_matrix dim1 dim2 2.0)) in
let e = Tensor.from_array (Tensor.A2D (Array.make_matrix dim2 dim3 3.0)) in

let f = Tensor.matmul d e in 
match f with
| T2D f2D -> begin 
   List.iter (fun i -> 
       List.iter (fun j ->  
        Printf.printf "%f\t" f2D.{i, j}) 
      [0; 1]) 
        [0; 1; 2; 3]
end 
| _ -> print_endline "Error"

