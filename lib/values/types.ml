open Bigarray

exception TypeException
exception SizeException

type standard_array = float array array 

type values = (float, float32_elt, c_layout) Array2.t

type dimensions = int * int
