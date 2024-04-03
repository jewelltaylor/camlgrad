open Bigarray
exception InvalidArgumentException

let range start stop =
 if start >= stop then raise InvalidArgumentException;

 let rec rangeHelper start stop =
   if start == stop then []
   else start :: (rangeHelper (start+1) stop) 
 in rangeHelper start stop

let dims a = (Array2.dim1 a, Array2.dim2 a)

let print_row a i =
  List.iter (fun j -> Printf.printf "%f  " a.{i, j}) (range 0 (Array2.dim2 a));
  Printf.printf "\n"

let print a = List.iter (fun i -> print_row a i) (range 0 (Array2.dim1 a))
