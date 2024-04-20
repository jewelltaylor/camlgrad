open Values

exception InvalidArgumentException of string
exception SizeException of string

type gradient = 
  | GRAD of values
  | NONE

type tensor = {
  tid : int;
  mutable vals : values;
  mutable grad : gradient;
  mutable acc_grad : gradient;
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
  | RELU of tensor
  | SIGMOID of tensor
  | CREATE
