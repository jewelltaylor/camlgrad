open Bigarray
open Ctypes
open Vforce

open TensorTypes

let unary_op1D f a = 
  match a with
  | T1D a1D -> begin 
    let dim = Array1.dim a1D in 
    let y1D = Array1.create Float32 c_layout dim in
    f (bigarray_start array1 y1D) (bigarray_start array1 a1D) (allocate int dim);
    T1D y1D
  end
  | _ -> raise TypeException

let unary_op2D f a = 
  match a with 
  | T2D a2D -> begin 
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in 
    f (bigarray_start array2 y2D) (bigarray_start array2 a2D) (allocate int (dim1 * dim2));
    T2D y2D
  end
  | _ -> raise TypeException
let unary_op f a =
  match a with
  | T1D _ -> unary_op1D f a
  | T2D _ -> unary_op2D f a

let log a = unary_op vvlogf a
let exp a = unary_op vvexpf a

