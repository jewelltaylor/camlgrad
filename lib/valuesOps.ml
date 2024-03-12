open Bigarray
open Ctypes
open Vforce
open Cblas

open Types 

let unary_op1D f a = 
  match a with
  | V1D a1D -> begin 
    let dim = Array1.dim a1D in 
    let y1D = Array1.create Float32 c_layout dim in
    f (bigarray_start array1 y1D) (bigarray_start array1 a1D) (allocate int dim);
    V1D y1D
  end
  | _ -> raise TypeException

let unary_op2D f a = 
  match a with 
  | V2D a2D -> begin 
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in 
    f (bigarray_start array2 y2D) (bigarray_start array2 a2D) (allocate int (dim1 * dim2));
    V2D y2D
  end
  | _ -> raise TypeException

let unary_op f a =
  match a with
  | V1D _ -> unary_op1D f a
  | V2D _ -> unary_op2D f a

let log a = unary_op vvlogf a
let exp a = unary_op vvexpf a
let sqrt a = unary_op vvsqrtf a
let reciprocal a = unary_op vvrecf a

let vforce_elementwise_binary_op1D f a b = 
  match (a, b) with
  | (V1D a1D, V1D b1D) -> begin
    if Array1.dim a1D <> Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    let y1D = Array1.create Float32 c_layout dim in  
    f (bigarray_start array1 y1D) (bigarray_start array1 a1D) (bigarray_start array1 b1D) (allocate int dim);
    V1D y1D
  end
  | _ -> raise TypeException

let vforce_elementwise_binary_op2D f a b = 
  match (a, b) with
  | (V2D a2D, V2D b2D) -> begin
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in  
    f (bigarray_start array2 y2D) (bigarray_start array2 a2D) (bigarray_start array2 b2D) (allocate int (dim1 * dim2));
    V2D y2D
  end
  | _ -> raise TypeException

let vforce_elementwise_binary_op f a b =
  match (a, b) with
  | (V1D _, V1D _) -> vforce_elementwise_binary_op1D f a b
  | (V2D _, V2D _) -> vforce_elementwise_binary_op2D f a b
  | _ -> raise TypeException

let pow2 a = 
  match a with
  | V1D a1D -> begin
    let dim = Array1.dim a1D in
    let x1D = Array1.init Float32 c_layout dim (fun _ -> 2.0) in 
    vforce_elementwise_binary_op vvpowf (V1D x1D) a 
  end
  | V2D a2D -> begin 
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let x2D = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> 2.0) in
    vforce_elementwise_binary_op vvpowf (V2D x2D) a
  end

let dotV1D a b = 
  match (a, b) with
  | (V1D a1D, V1D b1D) -> begin
    if Array1.dim a1D <> Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    cblas_sdot dim (bigarray_start array1 a1D) 1 (bigarray_start array1 b1D) 1
  end
  | _ -> raise TypeException 

let dotV2D a b = 
  match (a, b) with 
  | (V2D a2D, V2D b2D) -> begin
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    cblas_sdot (dim1 * dim2) (bigarray_start array2 a2D) 1 (bigarray_start array2 b2D) 1
  end
  | _ -> raise TypeException 

let dot a b = 
  match (a, b) with 
  | (V1D _, V1D _) -> dotV1D a b
  | (V2D _, V2D _) -> dotV2D a b
  | _ -> raise TypeException 

let sumV1D a = 
  match a with
  | V1D a1D -> begin
    let dim = Array1.dim a1D in
    let x1D = Array1.init Float32 c_layout dim (fun _ -> 1.0) in
    dotV1D a (V1D x1D)   
  end
  | _ -> raise SizeException

let sumV2D a = 
  match a with 
  | V2D a2D -> begin 
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let x2D = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> 1.0) in 
    dotV2D a (V2D x2D)
  end
  | _ -> raise SizeException

let sum a = 
  match a with
  | V1D _ -> sumV1D a 
  | V2D _ -> sumV2D a 

let div a b = vforce_elementwise_binary_op vvdivf a b

let mul a b =
  match (a, b) with
  | (V1D _, V1D _) -> div a (reciprocal b)
  | (V2D _, V2D _) -> div a (reciprocal b)
  | _ -> raise TypeException

let negV1D a =
  match a with
  | (V1D a1D) -> begin
    let dim = Array1.dim a1D in
    let x1D = Array1.init Float32 c_layout dim (fun _ -> -1.0) in
    mul a (V1D x1D)
  end
  | _ -> raise TypeException

let negV2D a =
  match a with
  | (V2D a2D) -> begin
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let x2D = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> -1.0) in 
    mul a (V2D x2D)
  end
  | _ -> raise TypeException

let neg a = 
  match a with
  | V1D _ -> negV1D a
  | V2D _ -> negV2D a

let addV1D a b =
  match (a, b) with
  | (V1D a1D, V1D b1D) -> begin 
    if Array1.dim a1D <> Array1.dim b1D then raise SizeException;
    let dim = Array1.dim a1D in
    let y1D = Array1.create Float32 c_layout dim in Array1.blit b1D y1D;
    cblas_saxpy dim 1.0 (bigarray_start array1 a1D) 1 (bigarray_start array1 y1D) 1;
    V1D y1D 
  end
  | _ -> raise TypeException 

let addV2D a b =
  match (a, b) with
  | (V2D a2D, V2D b2D) -> begin 
    if (Array2.dim1 a2D, Array2.dim2 a2D) <> (Array2.dim1 b2D, Array2.dim2 b2D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y2D = Array2.create Float32 c_layout dim1 dim2 in Array2.blit b2D y2D; 
    cblas_saxpy (dim1 * dim2) 1.0 (bigarray_start array2 a2D) 1 (bigarray_start array2 y2D) 1;
    V2D y2D 
  end
  | _ -> raise TypeException 

let add a b = 
  match (a, b) with
  | (V1D _, V1D _) -> addV1D a b
  | (V2D _, V2D _) -> addV2D a b
  | _ -> raise TypeException 

let matvecmul ?(trans_a = 111) a b = 
  match (a, b) with 
  | (V2D a2D, V1D b1D) -> begin
    if (Array2.dim2 a2D <> Array1.dim b1D) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a2D, Array2.dim2 a2D) in
    let y1D = Array1.init Float32 c_layout dim1 (fun _ -> 0.0) in 
    cblas_sgemv 101 trans_a dim1 dim2 1.0 (bigarray_start array2 a2D) dim2
      (bigarray_start array1 b1D) 1 0.0 (bigarray_start array1 y1D) 1;
    V1D y1D
  end
  | _ -> raise TypeException 

let matmul ?(trans_a = 111) ?(trans_b = 111) a b = 
  match (a, b) with 
  | (V2D a2D, V2D b2D) -> begin
    if (Array2.dim2 a2D <> Array2.dim1 b2D) then raise SizeException;
    let (adim1, adim2, _, bdim2) = 
      (Array2.dim1 a2D, Array2.dim2 a2D, Array2.dim1 b2D, Array2.dim2 b2D) in
    let y2D = Array2.init Float32 c_layout adim1 bdim2 (fun _ _ -> 0.0) in 
    cblas_sgemm 101 trans_a trans_b adim1 bdim2 adim2 1.0 (bigarray_start array2 a2D) 
      adim2 (bigarray_start array2 b2D) bdim2 0.0 (bigarray_start array2 y2D) bdim2;
    V2D y2D
  end
  | _ -> raise TypeException 

let sub a b = 
  match (a, b) with 
  | (V1D _, V1D _) -> add a (neg b)
  | (V2D _, V2D _) -> add a (neg b)
  | _ -> raise TypeException

let dim a =
  match a with
  | V1D a1D -> D1D (Array1.dim a1D)
  | V2D a2D -> D2D (Array2.dim1 a2D, Array2.dim2 a2D)
