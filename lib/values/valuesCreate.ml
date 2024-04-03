open Bigarray
open Types

let from_array (arr : standard_array) = Array2.of_array Float32 c_layout arr 

let create (dims: dimensions) (value : float) = 
  let (dim1, dim2) = dims in 
   Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> value)

let ones (dims: dimensions) = create dims 1.0 
let zeros (dims : dimensions) = create dims 0.0

let random (dims : dimensions) = 
  let (dim1, dim2) = dims in 
   Array2.init Float32 c_layout dim1 dim2 (fun _ _ -> (Random.float 1.0) -. 0.5 )

