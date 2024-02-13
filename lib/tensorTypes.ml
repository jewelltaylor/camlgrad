open Bigarray

exception TypeException
exception SizeException
exception InvalidArgumentException

type standard_array = 
  | A1D of float array
  | A2D of float array array 

type tensor = 
  | T1D of (float, float32_elt, c_layout) Array1.t
  | T2D of (float, float32_elt, c_layout) Array2.t

let from_array (arr : standard_array) = 
  match arr with 
  | (A1D a1D) -> T1D (Array1.of_array Float32 c_layout a1D)
  | (A2D a2D) -> T2D (Array2.of_array Float32 c_layout a2D) 
