
let mlp = NeuralNet.get_mlp [|(Tensor.sigmoid, (4, 1))|] in
let input = Tensor.from_array [|[|1.0; 2.0; 3.0; 4.0|]|] in
let (pred, _) = NeuralNet.mlp_forward mlp input in 
let target = Tensor.ones (1, 1) in
let bce_loss = NeuralNet.binary_cross_entropy pred target in
Tensor.backward bce_loss;
Tensor.printVals mlp.(0).weights;
Tensor.printVals mlp.(0).bias;
Tensor.printGrad mlp.(0).weights;
Tensor.printGrad mlp.(0).bias;
