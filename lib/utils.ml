open Bigarray 

open TensorTypes 

let range start stop =
 if start >= stop then raise InvalidArgumentException;

 let rec rangeHelper start stop =
   if start == stop then []
   else start :: (rangeHelper (start+1) stop) 
 in rangeHelper start stop

 let print1D a = 
   match a with
   | T1D a1D -> List.iter (fun i -> Printf.printf "%f\t" a1D.{i}) (range 0 (Array1.dim a1D))
   | _ -> raise TypeException

  let print2D a = 
    match a with 
    | T2D a2D -> begin
      List.iter (fun i -> 
        List.iter (fun j -> 
          Printf.printf "%f\t" a2D.{i, j}) 
        (range 0 (Array2.dim2 a2D)))
      (range 0 (Array2.dim1 a2D))
    end
    | _ -> raise TypeException
 
