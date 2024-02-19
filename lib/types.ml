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

type dimensions = 
  | D1D of int
  | D2D of int * int

type operator = ADD | SUB | MUL | DIV | MATMUL | MATVECMUL | NEG | EXP | LOG | SQRT | POW2 | CREATE

type tensor = {
  tid : int;
  vals : values;
  mutable grad : values;
  op : operator;
  children : tensor array
}

