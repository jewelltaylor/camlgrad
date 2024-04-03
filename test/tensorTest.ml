open TestUtils

let test_add () = 
  let a = Tensor.create (2, 4) 10.0 in
  let b = Tensor.create (2, 4) 10.0 in
  let c = Tensor.add a b in
  let c_gt = Tensor.create (2, 4) 20.0 in
  check_equal c.vals c_gt.vals;

  let r = Tensor.sum c in
  Tensor.backward r;

  let dr_da = Values.create (2, 4) 1.0 in
  let dr_db = Values.create (2, 4) 1.0 in
  check_equal dr_da (Tensor.get_grad a);
  check_equal dr_db (Tensor.get_grad b)

let test_sub () = 
  let a = Tensor.create (6, 4) 90.0 in
  let b = Tensor.create (6, 4) 10.0 in
  let c = Tensor.sub a b in
  let c_gt = Tensor.create (6, 4) 80.0 in
  check_equal c.vals c_gt.vals;

  let r = Tensor.sum c in
  Tensor.backward r;

  let dr_da = Values.create (6, 4) 1.0 in
  let dr_db = Values.create (6, 4) (-1.0) in
  check_equal dr_da (Tensor.get_grad a);
  check_equal dr_db (Tensor.get_grad b)

let test_mul () =
  let a = Tensor.create (6, 10) 1.5 in
  let b = Tensor.create (6, 10) 10.0 in
  let c = Tensor.mul a b in
  let c_gt = Tensor.create (6, 10) 15.0 in
  check_equal c.vals c_gt.vals;

  let r = Tensor.sum c in
  Tensor.backward r;

  let dr_da = Values.create (6, 10) 10.0 in
  let dr_db = Values.create (6, 10) (1.5) in
  check_equal dr_da (Tensor.get_grad a);
  check_equal dr_db (Tensor.get_grad b)

let test_div () =
  let a = Tensor.create (3, 3) 5.5 in
  let b = Tensor.create (3, 3) 0.5 in
  let c = Tensor.div a b in
  let c_gt = Tensor.create (3, 3) 11.0 in
  check_equal c.vals c_gt.vals;

  let r = Tensor.sum c in
  Tensor.backward r;

  let dr_da = Values.create (3, 3) 2.0 in
  let dr_db = Values.create (3, 3) (-22.0) in
  check_equal dr_da (Tensor.get_grad a);
  check_equal dr_db (Tensor.get_grad b)

let test_matmul () = 
  let a = Tensor.create (2, 4) 10.0 in
  let b = Tensor.create (4, 3) 15.0 in
  let c = Tensor.matmul a b in
  let c_gt = Tensor.create (2, 3) 600.0 in
  check_equal c.vals c_gt.vals;

  let r = Tensor.sum c in
  Tensor.backward r;
  let dr_da = Values.create (2, 4) 45.0 in
  let dr_db = Values.create (4, 3) 20.0 in
  check_equal dr_da (Tensor.get_grad a);
  check_equal dr_db (Tensor.get_grad b)

let test_neg () =
  let a = Tensor.create (10, 3) 103.3 in
  let b = Tensor.neg a in
  let b_gt = Tensor.create (10, 3) (-103.3) in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (10, 3) (-1.0) in
  check_equal dr_da (Tensor.get_grad a)

let test_exp () =
  let a = Tensor.create (10, 3) 4.0 in
  let b = Tensor.exp a in
  let b_gt = Tensor.create (10, 3) (54.5981) in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (10, 3) (54.5981) in
  check_equal dr_da (Tensor.get_grad a)

let test_log () =
  let a = Tensor.create (7, 30) 4.0 in
  let b = Tensor.log a in
  let b_gt = Tensor.create (7, 30) 1.38629 in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (7, 30) 0.25 in
  check_equal dr_da (Tensor.get_grad a)

let test_pow2 () =
  let a = Tensor.create (9, 10) 4.0 in
  let b = Tensor.pow2 a in
  let b_gt = Tensor.create (9, 10) 16.0 in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (9, 10) 8.0 in
  check_equal dr_da (Tensor.get_grad a)

let test_sqrt () =
  let a = Tensor.create (90, 10) 64.0 in
  let b = Tensor.sqrt a in
  let b_gt = Tensor.create (90, 10) 8.0 in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (90, 10) 0.0625 in
  check_equal dr_da (Tensor.get_grad a)

let test_relu () =
  let a = Tensor.from_array [|[|4.0; -4.0|]; [|-4.0; 4.0|]|] in
  let b = Tensor.relu a in
  let b_gt = Tensor.from_array [|[|4.0; 0.0|]; [|0.0; 4.0|]|] in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Tensor.from_array [|[|1.0; 0.0|]; [|0.0; 1.0|]|] in
  check_equal dr_da.vals (Tensor.get_grad a)

let test_sum () =
  let a = Tensor.create (5, 2) 10.0 in
  let b = Tensor.sum a in
  let b_gt = Tensor.create (1, 1) 100.0 in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (5, 2) 1.0 in
  check_equal dr_da (Tensor.get_grad a)

let test_sigmoid () =
  let a = Tensor.create (5, 5) 0.5 in
  let b = Tensor.sigmoid a in
  let b_gt = Tensor.create (5, 5) 0.622459 in
  check_equal b.vals b_gt.vals;

  let r = Tensor.sum b in
  Tensor.backward r;

  let dr_da = Values.create (5, 5) 0.2350 in
  check_equal dr_da (Tensor.get_grad a)

let () =
  let open Alcotest in
  run "Tensor Operations" [
    "addition", [test_case "Test add function" `Quick test_add];
    "subtract", [test_case "Test sub function" `Quick test_sub];
    "multiply", [test_case "Test mul function" `Quick test_mul];
    "divide", [test_case "Test div function" `Quick test_div];
    "negative", [test_case "Test neg function" `Quick test_neg];
    "exponential", [test_case "Test neg function" `Quick test_exp];
    "logarithm", [test_case "Test log function" `Quick test_log];
    "power", [test_case "Test pow2 function" `Quick test_pow2];
    "sqrt", [test_case "Test sqrt function" `Quick test_sqrt];
    "matmul", [test_case "Test matmul function" `Quick test_matmul];
    "relu", [test_case "Test relu function" `Quick test_relu];
    "sum", [test_case "Test sum function" `Quick test_sum];
    "sigmoid", [test_case "Test sigmoid function" `Quick test_sigmoid];
  ]
