open Bigarray

exception InvalidArgumentException of string
exception SizeException of string

type standard_array = float array array 

type values = (float, float32_elt, c_layout) Array2.t

type dimensions = int * int
