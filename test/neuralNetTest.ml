open Camlgrad
open Alcotest

exception SizeException

let check_equal a b =
  if (Values.dim a <> Values.dim b) then raise SizeException;
  let (d1, d2) = Values.dim a in
  List.iter (fun i -> 
    List.iter (fun j -> 
      let epsilon_float = 0.0001 in 
      check (float epsilon_float) "Equal Float" a.{i, j} b.{i, j}
  )(Utils.range 0 d2)) (Utils.range 0 d1)

let test_mean_squared_error () = 
  let pred = Tensor.create (4, 4) 2.0 in
  let target = Tensor.create (4, 4) 4.0 in
  let mse = NeuralNet.mean_squared_error pred target in
  let gt_mse = Tensor.create (1, 1) 4.0 in
  check_equal mse.vals gt_mse.vals

let test_binary_cross_entropy () =
  let pred = Tensor.from_array [|[|0.502243; 0.306248|]|] in
  let target = Tensor.from_array [|[|1.0; 0.0|]|] in
  let bce = NeuralNet.binary_cross_entropy pred target in
  let gt_bce = Tensor.create (1, 1) 0.527156 in
  check_equal bce.vals gt_bce.vals

let test_mlp_layer_forward () =
  let (dimIn, dimOut) = (2, 2) in
  let mlp_layer = NeuralNet.get_mlp_layer (dimIn, dimOut) in
  let input = Tensor.ones (1, 2) in
  let result = NeuralNet.mlp_layer_forward mlp_layer input in
  let gt_result = Tensor.from_array [|[|0.4538; 0.5554|]|] in
  check_equal result.vals gt_result.vals

let test_mlp_forward () =
  let (dimIn, dimOut) = (2, 2) in
  let mlp = NeuralNet.get_mlp [|(Tensor.sigmoid, (dimIn, dimOut)); (Tensor.sigmoid, (dimIn, dimOut))|] in
  let input = Tensor.ones (1, 2) in
  let (result, int_results) = NeuralNet.mlp_forward mlp input in
  let gt_int_results = [|Tensor.from_array [|[|0.645525; 0.466234|]|]; Tensor.from_array [|[|0.456988; 0.406843|]|]|] in 
  check_equal int_results.(0).vals gt_int_results.(0).vals;
  check_equal int_results.(1).vals gt_int_results.(1).vals;
  check_equal result.vals gt_int_results.(1).vals

let test_zero_grad_mlp_layer () =
  let mlp_layer = NeuralNet.get_mlp_layer (5, 5) in 
  let input = Tensor.ones (1, 5) in
  let pred = NeuralNet.mlp_layer_forward mlp_layer input in
  let target = Tensor.zeros (1, 5) in
  let bce_loss = NeuralNet.binary_cross_entropy pred target in
  Tensor.backward bce_loss;
  NeuralNet.apply_update mlp_layer 0.01; 
  NeuralNet.zero_grad_mlp_layer mlp_layer;
  check_equal (Tensor.get_grad mlp_layer.weights) (Values.zeros (5, 5));
  check_equal (Tensor.get_grad mlp_layer.bias) (Values.zeros (1, 5))

let test_zero_grad_mlp () = 
  let mlp = NeuralNet.get_mlp [|(Tensor.sigmoid, (4, 5)); (Tensor.sigmoid, (5, 4))|] in 
  let input = Tensor.random (1, 4) in
  let (pred, _) = NeuralNet.mlp_forward mlp input in
  let target = Tensor.zeros (1, 4) in
  let bce_loss = NeuralNet.binary_cross_entropy pred target in
  Tensor.backward bce_loss;
  NeuralNet.gradient_descent mlp bce_loss 0.01; 
  NeuralNet.zero_grad_mlp mlp;
  check_equal (Tensor.get_grad mlp.(0).weights) (Values.zeros (4, 5));
  check_equal (Tensor.get_grad mlp.(0).bias) (Values.zeros (1, 5));
  check_equal (Tensor.get_grad mlp.(1).weights) (Values.zeros (5, 4));
  check_equal (Tensor.get_grad mlp.(1).bias) (Values.zeros (1, 4))

let test_gradient_descent () =
  let mlp = NeuralNet.get_mlp [|(Tensor.sigmoid, (4, 1))|] in
  let input = Tensor.from_array [|[|1.0; 2.0; 3.0; 4.0|]|] in
  let (pred, _) = NeuralNet.mlp_forward mlp input in 
  let target = Tensor.ones (1, 1) in
  let bce_loss = NeuralNet.binary_cross_entropy pred target in
  let gt_weight_acc_grad = Values.from_array [|[|-0.1307|]; [|-0.2614|]; [|-0.3921|]; [|-0.5228|]|] in
  let gt_new_weight = Values.from_array [|[|0.1269|]; [|0.4377|]; [|-0.4152|]; [|0.5139|]|] in
  let gt_bias_acc_grad = Values.from_array [|[|-0.1307|]|] in
  let gt_new_bias = Values.from_array [|[|0.4876|]|] in
  NeuralNet.gradient_descent mlp bce_loss 0.1; 
  check_equal (Tensor.get_acc_grad mlp.(0).weights) gt_weight_acc_grad; 
  check_equal (Tensor.get_acc_grad mlp.(0).bias) gt_bias_acc_grad;
  check_equal mlp.(0).weights.vals gt_new_weight; 
  check_equal mlp.(0).bias.vals gt_new_bias 

let () =
  let open Alcotest in
  run "Neural Net Operations" [
    "mse", [test_case "Test mean squared error function" `Quick test_mean_squared_error];
    "bce", [test_case "Test binary cross entropy loss function" `Quick test_binary_cross_entropy];
    "mlp layer forward", [test_case "Test mlp layer forward function" `Quick test_mlp_layer_forward];
    "mlp forward", [test_case "Test mlp forward function" `Quick test_mlp_forward];
    "zero grad mlp layer", [test_case "Test zero grad mlp layer function" `Quick test_zero_grad_mlp_layer];
    "zero grad mlp", [test_case "Test zero grad mlp function" `Quick test_zero_grad_mlp];
    "gradient descent mlp", [test_case "Test gradient descent mlp function" `Quick test_gradient_descent];
  ]
