open Bigarray

exception TypeException
exception SizeException
exception InvalidArgumentException

type standard_array = float array array 

type values = (float, float32_elt, c_layout) Array2.t

type dimensions = int * int

type gradient = 
  | GRAD of values
  | NONE

type tensor = {
  tid : int;
  vals : values;
  mutable grad : gradient;
  op : operator;
} and operator = 
  | ADD of tensor * tensor 
  | SUB of tensor * tensor 
  | MUL of tensor * tensor 
  | DIV of tensor * tensor
  | MATMUL of tensor * tensor
  | NEG of tensor 
  | EXP of tensor 
  | LOG of tensor 
  | SQRT of tensor
  | POW2 of tensor 
  | SUM of tensor 
  | CREATE

