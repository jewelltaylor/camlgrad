
let layer_dim_arr = [|(5, 3); (3, 2)|] in
let layer_spec_arr = Array.map (fun dim -> (Tensor.sigmoid, dim)) layer_dim_arr in 

let mlp = NeuralNet.get_mlp layer_spec_arr in
let input = Tensor.random (1, 5) in
let (final_result, _) = NeuralNet.mlp_forward mlp input in
let gt_result = Tensor.from_array [|[|1.0; 0.0|]|] in 
let bce_loss = NeuralNet.binary_cross_entropy final_result gt_result in

Tensor.printVals final_result;
Tensor.printVals bce_loss;
