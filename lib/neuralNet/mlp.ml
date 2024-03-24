open Camlgrad.Types

let get_mlp_layer ?(activation = Tensor.sigmoid) dims =
  let (_, d2) = dims in
  let weights = Tensor.random dims in 
  let bias = Tensor.random (1, d2) in
  {weights; bias; activation}

let mlp_layer_forward layer input =
  let linearTransform = Tensor.matmul input layer.weights in
  let affineTransform = Tensor.add linearTransform layer.bias in 
  layer.activation affineTransform 

let get_mlp layer_spec_arr =
  Array.map (fun (act, dim) -> get_mlp_layer ~activation:act dim) layer_spec_arr 

let mlp_forward mlp input = Array.fold_left_map (fun acc mlp_layer -> 
  let result = mlp_layer_forward mlp_layer acc in
  (result, result)
  ) input mlp
