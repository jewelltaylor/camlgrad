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
  let gt_result = Tensor.from_array [|[|0.593148; 0.344535|]|] in
  check_equal result.vals gt_result.vals

let test_mlp_forward () =
  let (dimIn, dimOut) = (2, 2) in
  let mlp = NeuralNet.get_mlp [|(Tensor.sigmoid, (dimIn, dimOut)); (Tensor.sigmoid, (dimIn, dimOut))|] in
  let input = Tensor.ones (1, 2) in
  let (result, int_results) = NeuralNet.mlp_forward mlp input in
  let gt_int_results = [|Tensor.from_array [|[|0.261923; 0.55867|]|]; Tensor.from_array [|[|0.4209009; 0.38563|]|]|] in 
  check_equal int_results.(0).vals gt_int_results.(0).vals;
  check_equal int_results.(1).vals gt_int_results.(1).vals;
  check_equal result.vals gt_int_results.(1).vals

let () =
  let open Alcotest in
  run "Tensor Operations" [
    "mse", [test_case "Test mean squared error function" `Quick test_mean_squared_error];
    "bce", [test_case "Test binary cross entropy loss function" `Quick test_binary_cross_entropy];
    "mlp layer forward", [test_case "Test mlp layer forward function" `Quick test_mlp_layer_forward];
    "mlp forward", [test_case "Test mlp forward function" `Quick test_mlp_forward];
  ]
