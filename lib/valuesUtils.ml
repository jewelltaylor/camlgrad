open Bigarray

open Types
open Utils


 let print1D a = 
   match a with
   | V1D a1D -> List.iter (fun i -> Printf.printf "%f\t" a1D.{i}) (range 0 (Array1.dim a1D))
   | _ -> raise TypeException

  let print2D a = 
    match a with 
    | V2D a2D -> begin
      List.iter (fun i -> 
        List.iter (fun j -> 
          Printf.printf "%f\t" a2D.{i, j}) 
        (range 0 (Array2.dim2 a2D)))
      (range 0 (Array2.dim1 a2D))
    end
    | _ -> raise TypeException

  let print a = 
    match a with 
    | V1D _ -> print1D a
    | V2D _ -> print2D a 

let dims a = 
  match a with 
  | V1D a1D -> D1D (Array1.dim a1D)
  | V2D a2D -> D2D (Array2.dim1 a2D, Array2.dim2 a2D)

