open Bigarray

open Types

let from_array (arr : standard_array) = 
  match arr with 
  | (A1D a1D) -> V1D (Array1.of_array Float32 c_layout a1D)
  | (A2D a2D) -> V2D (Array2.of_array Float32 c_layout a2D) 
