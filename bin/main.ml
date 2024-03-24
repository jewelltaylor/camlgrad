
let layer_dim_arr = [|(5, 3); (3, 9)|] in
let layer_spec_arr = Array.map (fun dim -> (Tensor.sigmoid, dim)) layer_dim_arr in 

let mlp = NeuralNet.get_mlp layer_spec_arr in
let input = Tensor.random (1, 5) in
let (final_result, _) = NeuralNet.mlp_forward mlp input in

Tensor.printVals input;
Tensor.printVals final_result;
