
let layer_dim_arr = [|(5, 3); (3, 2)|] in
let layer_spec_arr = Array.map (fun dim -> (Tensor.sigmoid, dim)) layer_dim_arr in 

let mlp = NeuralNet.get_mlp layer_spec_arr in
let input = Tensor.random (1, 5) in
let (pred1, _) = NeuralNet.mlp_forward mlp input in
let gt_result = Tensor.from_array [|[|1.0; 0.0|]|] in 
let bce_loss = NeuralNet.binary_cross_entropy pred1 gt_result in
NeuralNet.stochastic_gradient_descent mlp bce_loss 0.01;
let (pred2, _) = NeuralNet.mlp_forward mlp input in
let bce_loss2 = NeuralNet.binary_cross_entropy pred2 gt_result in
Tensor.printVals bce_loss;
Tensor.printVals bce_loss2;
