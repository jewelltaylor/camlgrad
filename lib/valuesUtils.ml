open Bigarray

open Utils


  let print a = 
      List.iter (fun i -> 
        List.iter (fun j -> 
          Printf.printf "%f\t" a.{i, j}) 
        (range 0 (Array2.dim2 a)))
      (range 0 (Array2.dim1 a))

let dims a = (Array2.dim1 a, Array2.dim2 a)

