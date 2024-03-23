open Bigarray
open Ctypes
open Camlgrad.Vforce
open Camlgrad.Cblas
open Camlgrad.Types

let unary_op f a = 
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let y = Array2.create Float32 c_layout dim1 dim2 in 
    f (bigarray_start array2 y) (bigarray_start array2 a) (allocate int (dim1 * dim2));
    y 

let log a = unary_op vvlogf a
let exp a = unary_op vvexpf a
let sqrt a = unary_op vvsqrtf a
let reciprocal a = unary_op vvrecf a

let vforce_elementwise_binary_op f a b = 
    if (Array2.dim1 a, Array2.dim2 a) <> (Array2.dim1 b, Array2.dim2 b) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let y = Array2.create Float32 c_layout dim1 dim2 in  
    f (bigarray_start array2 y) (bigarray_start array2 a) (bigarray_start array2 b) (allocate int (dim1 * dim2));
    y 

let pow2 a = 
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let x = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> 2.0) in
    vforce_elementwise_binary_op vvpowf x a

let dot a b = 
    if (Array2.dim1 a, Array2.dim2 a) <> (Array2.dim1 b, Array2.dim2 b) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let r = cblas_sdot (dim1 * dim2) (bigarray_start array2 a) 1 (bigarray_start array2 b) 1 in
    Array2.init Float32 c_layout 1 1 (fun _ _ -> r)

let sum a = 
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let x = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> 1.0) in 
    dot a x 

let div a b = vforce_elementwise_binary_op vvdivf a b
let mul a b = div a (reciprocal b)

let neg a =
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let x = Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> -1.0) in 
    mul a x 

let add a b =
    if (Array2.dim1 a, Array2.dim2 a) <> (Array2.dim1 b, Array2.dim2 b) then raise SizeException;
    let (dim1, dim2) = (Array2.dim1 a, Array2.dim2 a) in
    let y = Array2.create Float32 c_layout dim1 dim2 in Array2.blit b y; 
    cblas_saxpy (dim1 * dim2) 1.0 (bigarray_start array2 a) 1 (bigarray_start array2 y) 1;
    y 

let matmul ?(trans_a = 111) ?(trans_b = 111) a b = 
    match (trans_a, trans_b) with
    | (111, 111) -> begin
        if (Array2.dim2 a <> Array2.dim1 b) then raise SizeException;
        let adim1, adim2, bdim2 = Array2.dim1 a, Array2.dim2 a, Array2.dim2 b in
        let y = Array2.init Float32 c_layout adim1 bdim2 (fun _ _ -> 0.0) in 
        cblas_sgemm 101 trans_a trans_b adim1 bdim2 adim2 1.0 (bigarray_start array2 a) 
          adim2 (bigarray_start array2 b) bdim2 0.0 (bigarray_start array2 y) bdim2;
        y 
    end
    | (111, 112) -> begin
        if (Array2.dim2 a <> Array2.dim2 b) then raise SizeException;
        let adim1, adim2, bdim1, bdim2 = Array2.dim1 a, Array2.dim2 a, Array2.dim1 b, Array2.dim2 b in
        let y = Array2.init Float32 c_layout adim1 bdim1 (fun _ _ -> 0.0) in 
        cblas_sgemm 101 trans_a trans_b adim1 bdim1 adim2 1.0 (bigarray_start array2 a) 
          adim2 (bigarray_start array2 b) bdim2 0.0 (bigarray_start array2 y) bdim1;
        y 
    end
    | (112, 111) -> begin
        if (Array2.dim1 a <> Array2.dim1 b) then raise SizeException;
        let adim1, adim2, _, bdim2 = Array2.dim1 a, Array2.dim2 a, Array2.dim1 b, Array2.dim2 b in
        let y = Array2.init Float32 c_layout adim2 bdim2 (fun _ _ -> 0.0) in 
        cblas_sgemm 101 trans_a trans_b adim2 bdim2 adim1 1.0 (bigarray_start array2 a) 
          adim2 (bigarray_start array2 b) bdim2 0.0 (bigarray_start array2 y) bdim2;
        y 
    end
    | _ -> raise TypeException


let sub a b = add a (neg b) 

let dim a = (Array2.dim1 a, Array2.dim2 a)
