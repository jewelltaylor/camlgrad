open Alcotest
open Camlgrad

exception SizeException

let check_equal a b =
  if (Values.dim a <> Values.dim b) then raise SizeException;
  let (d1, d2) = Values.dim a in
  List.iter (fun i -> 
    List.iter (fun j -> 
      let epsilon_float = 0.0001 in 
      check (float epsilon_float) "Equal Float" a.{i, j} b.{i, j}
  )(Utils.range 0 d2)) (Utils.range 0 d1)

let test_add () =
  let a = Values.create (5, 6) 15.0 in
  let b = Values.create (5, 6) 10.0 in
  let c = Values.add a b in
  let c_gt = Values.create (5, 6) 25.0 in
  check_equal c c_gt

let test_mul () =
  let a = Values.create (2, 3) 2.0 in
  let b = Values.create (2, 3) 35.0 in
  let c = Values.mul a b in
  let c_gt = Values.create (2, 3) 70.0 in
  check_equal c c_gt

let test_div () =
  let a = Values.create (8, 3) 22.0 in
  let b = Values.create (8, 3) 11.0 in
  let c = Values.div a b in
  let c_gt = Values.create (8, 3) 2.0 in
  check_equal c c_gt

let test_sub () =
  let a = Values.create (3, 4) 2.0 in
  let b = Values.create (3, 4) 35.0 in
  let c = Values.sub a b in
  let c_gt = Values.create (3, 4) (-33.0) in
  check_equal c c_gt

let test_neg () =
  let a = Values.create (5, 6) 33.0 in
  let b = Values.neg a in
  let b_gt = Values.create (5, 6) (-33.0) in
  check_equal b b_gt 

let test_log () =
  let a = Values.create (6, 3) 100.0 in
  let b = Values.log a in
  let b_gt = Values.create (6, 3) 4.60517 in
  check_equal b b_gt 

let test_pow2 () =
  let a = Values.create (20, 20) 10.0 in
  let b = Values.pow2 a in
  let b_gt = Values.create (20, 20) 100.0 in
  check_equal b b_gt 

let test_exp () =
  let a = Values.create (10, 20) 4.0 in
  let b = Values.exp a in
  let b_gt = Values.create (10, 20) 54.5981 in
  check_equal b b_gt 

let test_sqrt () =
  let a = Values.create (10, 50) 4.0 in
  let b = Values.sqrt a in
  let b_gt = Values.create (10, 50) 2.0 in
  check_equal b b_gt 

let test_sum () =
  let a = Values.create (5, 5) 4.0 in
  let b = Values.sum a in
  let b_gt = Values.create (1, 1) 100.0 in
  check_equal b b_gt 

let test_dot () =
  let a = Values.create (5, 5) 10.0 in
  let b = Values.create (5, 5) 10.0 in
  let c = Values.dot a b in
  let c_gt = Values.create (1, 1) 2500.0 in
  check_equal c c_gt 

let test_matmul () =
  let a = Values.from_array [|[|1.0; 2.0|]; [|3.0; 4.0|]|] in
  let b = Values.from_array [|[|1.0; 2.0|]; [|3.0; 4.0|]|] in
  let c = Values.matmul a b in
  let c_gt = Values.from_array [|[|7.0; 10.0|]; [|15.0; 22.0|]|] in
  check_equal c c_gt

let () =
  let open Alcotest in
  run "Value Operations" [
    "addition", [test_case "Test add function" `Quick test_add];
    "multiply", [test_case "Test multiply function" `Quick test_mul];
    "subtract", [test_case "Test subtract function" `Quick test_sub];
    "div", [test_case "Test divide function" `Quick test_div];
    "negative", [test_case "Test negative function" `Quick test_neg];
    "log", [test_case "Test log function" `Quick test_log];
    "pow2", [test_case "Test pow2 function" `Quick test_pow2];
    "exp", [test_case "Test exp function" `Quick test_exp];
    "sqrt", [test_case "Test sqrt function" `Quick test_sqrt];
    "matmul", [test_case "Test matmul function" `Quick test_matmul];
    "sum", [test_case "Test sum function" `Quick test_sum];
    "dot", [test_case "Test dot function" `Quick test_dot];
  ]

