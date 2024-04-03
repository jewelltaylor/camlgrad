open Camlgrad
open TestUtils

let test_mean_squared_error () = 
  let pred = Tensor.create (4, 4) 2.0 in
  let target = Tensor.create (4, 4) 4.0 in
  let mse = Loss.mean_squared_error pred target in
  let gt_mse = Tensor.create (1, 1) 4.0 in
  check_equal mse.vals gt_mse.vals

let test_binary_cross_entropy () =
  let pred = Tensor.from_array [|[|0.502243; 0.306248|]|] in
  let target = Tensor.from_array [|[|1.0; 0.0|]|] in
  let bce = Loss.binary_cross_entropy pred target in
  let gt_bce = Tensor.create (1, 1) 0.527156 in
  check_equal bce.vals gt_bce.vals

let () =
  let open Alcotest in
  run "Loss Functions" [
    "mse", [test_case "Test mean squared error function" `Quick test_mean_squared_error];
    "bce", [test_case "Test binary cross entropy loss function" `Quick test_binary_cross_entropy];
  ]
