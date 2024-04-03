open Camlgrad
open TestUtils 

let test_mlp_layer_forward () =
  let (dimIn, dimOut) = (2, 2) in
  let mlp_layer = Mlp.get_mlp_layer (dimIn, dimOut) in
  let input = Tensor.ones (1, 2) in
  let result = Mlp.mlp_layer_forward mlp_layer input in
  let gt_result = Tensor.from_array [|[|0.52155; 0.354303|]|] in
  check_equal result.vals gt_result.vals

let test_mlp_forward () =
  let (dimIn, dimOut) = (2, 2) in
  let mlp = Mlp.get_mlp [|(Tensor.sigmoid, (dimIn, dimOut)); (Tensor.sigmoid, (dimIn, dimOut))|] in
  let input = Tensor.ones (1, 2) in
  let (result, int_results) = Mlp.mlp_forward mlp input in
  let gt_int_results = [|Tensor.from_array [|[|0.305450; 0.297930|]|]; Tensor.from_array [|[|0.512771; 0.561868|]|]|] in 
  check_equal int_results.(0).vals gt_int_results.(0).vals;
  check_equal int_results.(1).vals gt_int_results.(1).vals;
  check_equal result.vals gt_int_results.(1).vals

let () =
  let open Alcotest in
  run "MLP Operations" [
    "mlp layer forward", [test_case "Test mlp layer forward function" `Quick test_mlp_layer_forward];
    "mlp forward", [test_case "Test mlp forward function" `Quick test_mlp_forward];
  ]
