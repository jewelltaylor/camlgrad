open Bigarray

exception TypeException
exception SizeException
exception InvalidArgumentException

type standard_array = 
  | A1D of float array
  | A2D of float array array 

type values = 
  | V1D of (float, float32_elt, c_layout) Array1.t
  | V2D of (float, float32_elt, c_layout) Array2.t

