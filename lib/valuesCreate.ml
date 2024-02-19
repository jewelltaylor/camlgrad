open Bigarray

open Types

let from_array (arr : standard_array) = 
  match arr with 
  | A1D a1D -> V1D (Array1.of_array Float32 c_layout a1D)
  | A2D a2D -> V2D (Array2.of_array Float32 c_layout a2D) 

let create (dims: dimensions) (value : float) = 
  match dims with
  | D1D dim -> V1D (Array1.init Float32 c_layout dim (fun _ -> value))
  | D2D (dim1, dim2) -> V2D (Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> value))

let ones (dims: dimensions) = create dims 1.0 
let zeros (dims : dimensions) = create dims 0.0
