open Camlgrad
open TestUtils

let test_zero_grad_mlp_layer () =
  let mlp_layer = Mlp.get_mlp_layer (5, 5) in 
  let input = Tensor.ones (1, 5) in
  let pred = Mlp.mlp_layer_forward mlp_layer input in
  let target = Tensor.zeros (1, 5) in
  let bce_loss = Loss.binary_cross_entropy pred target in
  Tensor.backward bce_loss;
  Optimizer.apply_update mlp_layer 0.01; 
  Optimizer.zero_grad_mlp_layer mlp_layer;
  check_equal (Tensor.get_grad mlp_layer.weights) (Values.zeros (5, 5));
  check_equal (Tensor.get_grad mlp_layer.bias) (Values.zeros (1, 5))

let test_zero_grad_mlp () = 
  let mlp = Mlp.get_mlp [|(Tensor.sigmoid, (4, 5)); (Tensor.sigmoid, (5, 4))|] in 
  let input = Tensor.random (1, 4) in
  let (pred, _) = Mlp.mlp_forward mlp input in
  let target = Tensor.zeros (1, 4) in
  let bce_loss = Loss.binary_cross_entropy pred target in
  Tensor.backward bce_loss;
  Optimizer.gradient_descent mlp bce_loss 0.01; 
  Optimizer.zero_grad_mlp mlp;
  check_equal (Tensor.get_grad mlp.(0).weights) (Values.zeros (4, 5));
  check_equal (Tensor.get_grad mlp.(0).bias) (Values.zeros (1, 5));
  check_equal (Tensor.get_grad mlp.(1).weights) (Values.zeros (5, 4));
  check_equal (Tensor.get_grad mlp.(1).bias) (Values.zeros (1, 4))

let test_gradient_descent () =
  let mlp = Mlp.get_mlp [|(Tensor.sigmoid, (4, 1))|] in
  let input = Tensor.from_array [|[|1.0; 2.0; 3.0; 4.0|]|] in
  let (pred, _) = Mlp.mlp_forward mlp input in 
  let target = Tensor.ones (1, 1) in
  let bce_loss = Loss.binary_cross_entropy pred target in
  let gt_weight_acc_grad = Values.from_array [|[|-0.797098|]; [|-1.594195|]; [|-2.391293|]; [|-3.188391|]|] in
  let gt_new_weight = Values.from_array [|[|-0.413116|]; [|-0.119047|]; [|0.329159|]; [|0.153281|]|] in
  let gt_bias_acc_grad = Values.from_array [|[|-0.797098|]|] in
  let gt_new_bias = Values.from_array [|[|0.153361|]|] in
  Optimizer.gradient_descent mlp bce_loss 0.1; 
  Tensor.printVals mlp.(0).bias;
  check_equal (Tensor.get_acc_grad mlp.(0).weights) gt_weight_acc_grad; 
  check_equal (Tensor.get_acc_grad mlp.(0).bias) gt_bias_acc_grad;
  check_equal mlp.(0).weights.vals gt_new_weight; 
  check_equal mlp.(0).bias.vals gt_new_bias 

let () =
  let open Alcotest in
  run "Optimizer Operations" [
    "zero grad mlp layer", [test_case "Test zero grad mlp layer function" `Quick test_zero_grad_mlp_layer];
    "zero grad mlp", [test_case "Test zero grad mlp function" `Quick test_zero_grad_mlp];
    "gradient descent mlp", [test_case "Test gradient descent mlp function" `Quick test_gradient_descent];
  ]
