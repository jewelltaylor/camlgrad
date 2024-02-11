open Bigarray
open Cblas
open Ctypes

exception SizeException

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

let addT1D a b =
  match (a, b) with
  | (T1D a1D, T1D b1D) -> begin 
    if Array1.dim a1D != Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    let y1D = Array1.create Float32 c_layout dim in Array1.blit b1D y1D;
    cblas_saxpy dim 1.0 (bigarray_start array1 a1D) 1 (bigarray_start array1 y1D) 1;
    T1D y1D 
  end
  | _ -> raise SizeException

let addT2D a b =
  match (a, b) with
  | (T2D a2D, T2D b2D) -> begin 
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in Array2.blit b2D y2D; 
    cblas_saxpy (dim1 * dim2) 1.0 (bigarray_start array2 a2D) 1 (bigarray_start array2 y2D) 1;
    T2D y2D 
  end
  | _ -> raise SizeException

let add a b = 
  match (a, b) with
  | (T1D _, T1D _) -> addT1D a b
  | (T2D _, T2D _) -> addT2D a b
  | _ -> raise SizeException

let dotT1D a b = 
  match (a, b) with
  | (T1D a1D, T1D b1D) -> begin
    if Array1.dim a1D != Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    cblas_sdot dim (bigarray_start array1 a1D) 1 (bigarray_start array1 b1D) 1
  end
  | _ -> raise SizeException

let dotT2D a b = 
  match (a, b) with 
  | (T2D a2D, T2D b2D) -> begin
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    cblas_sdot (dim1 * dim2) (bigarray_start array2 a2D) 1 (bigarray_start array2 b2D) 1
  end
  | _ -> raise SizeException

let dot a b = 
  match (a, b) with 
  | (T1D _, T1D _) -> dotT1D a b
  | (T2D _, T2D _) -> dotT2D a b
  | _ -> raise SizeException

