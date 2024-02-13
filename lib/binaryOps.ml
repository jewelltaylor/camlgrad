open Bigarray
open Cblas
open Ctypes

open TensorTypes


let addT1D a b =
  match (a, b) with
  | (T1D a1D, T1D b1D) -> begin 
    if Array1.dim a1D <> Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    let y1D = Array1.create Float32 c_layout dim in Array1.blit b1D y1D;
    cblas_saxpy dim 1.0 (bigarray_start array1 a1D) 1 (bigarray_start array1 y1D) 1;
    T1D y1D 
  end
  | _ -> raise TypeException 

let addT2D a b =
  match (a, b) with
  | (T2D a2D, T2D b2D) -> begin 
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in Array2.blit b2D y2D; 
    cblas_saxpy (dim1 * dim2) 1.0 (bigarray_start array2 a2D) 1 (bigarray_start array2 y2D) 1;
    T2D y2D 
  end
  | _ -> raise TypeException 

let add a b = 
  match (a, b) with
  | (T1D _, T1D _) -> addT1D a b
  | (T2D _, T2D _) -> addT2D a b
  | _ -> raise TypeException 

let dotT1D a b = 
  match (a, b) with
  | (T1D a1D, T1D b1D) -> begin
    if Array1.dim a1D <> Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    cblas_sdot dim (bigarray_start array1 a1D) 1 (bigarray_start array1 b1D) 1
  end
  | _ -> raise TypeException 

let dotT2D a b = 
  match (a, b) with 
  | (T2D a2D, T2D b2D) -> begin
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    cblas_sdot (dim1 * dim2) (bigarray_start array2 a2D) 1 (bigarray_start array2 b2D) 1
  end
  | _ -> raise TypeException 

let dot a b = 
  match (a, b) with 
  | (T1D _, T1D _) -> dotT1D a b
  | (T2D _, T2D _) -> dotT2D a b
  | _ -> raise TypeException 

let matvecmul a b = 
  match (a, b) with 
  | (T2D a2D, T1D b1D) -> begin
    if (Array2.dim2 a2D <> Array1.dim b1D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y1D = Array1.init Float32 c_layout dim1 (fun _ -> 0.0) in 
    cblas_sgemv 101 111 dim1 dim2 1.0 (bigarray_start array2 a2D) dim2
      (bigarray_start array1 b1D) 1 0.0 (bigarray_start array1 y1D) 1;
    T1D y1D
  end
  | _ -> raise TypeException 

let matmul a b = 
  match (a, b) with 
  | (T2D a2D, T2D b2D) -> begin
    if (Array2.dim2 a2D <> Array2.dim1 b2D) then raise SizeException;
    let (adim1, adim2, _, bdim2) = 
      (Array2.dim1 a2D, Array2.dim2 a2D, Array2.dim1 b2D, Array2.dim2 b2D) in
    let y2D = Array2.init Float32 c_layout adim1 bdim2 (fun _ _ -> 0.0) in 
    cblas_sgemm 101 111 111 adim1 bdim2 adim2 1.0 (bigarray_start array2 a2D) 
      adim2 (bigarray_start array2 b2D) bdim2 0.0 (bigarray_start array2 y2D) bdim2;
    T2D y2D
  end
  | _ -> raise TypeException 
