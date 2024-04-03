open Types
open Values
open TensorUtils
open TensorGraph

let accumulate_gradient a dims = 
  let acc_grad = get_grad_and_init_to_zero_if_none a.acc_grad dims in
  a.acc_grad <- GRAD (Values.add acc_grad (get_grad a))

let add_backward r = 
  match r.op with
  | ADD (a, b) -> begin 
    let r_grad = get_grad r in
    a.grad <- GRAD r_grad; 
    accumulate_gradient a (Values.dim r_grad);
    b.grad <- GRAD r_grad; 
    accumulate_gradient b (Values.dim r_grad);
  end
  | _ -> raise TypeException

let mul_backward r = 
  match r.op with
  | MUL (a, b) -> begin
    let r_grad = get_grad r in
    a.grad <- GRAD (Values.mul b.vals r_grad); 
    accumulate_gradient a (Values.dim r_grad); 
    b.grad <- GRAD (Values.mul a.vals r_grad); 
    accumulate_gradient b (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let div_backward r = 
  match r.op with
  | DIV (a, b) -> begin 
    let r_grad = get_grad r in
    a.grad <- GRAD (Values.div r_grad b.vals); 
    accumulate_gradient a (Values.dim r_grad); 
    b.grad <- GRAD (Values.mul r_grad (Values.neg (Values.div a.vals (Values.pow2 b.vals)))); 
    accumulate_gradient b (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let neg_backward r = 
  match r.op with
  | NEG a -> begin 
    let r_grad = get_grad r in
    let b = Values.create (Values.dim a.vals) (-1.0) in 
    a.grad <- GRAD (Values.mul b r_grad);
    accumulate_gradient a (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let sub_backward r = 
  match r.op with
  | SUB (a, b) -> begin 
    let r_grad = get_grad r in
    a.grad <- GRAD r_grad; 
    accumulate_gradient a (Values.dim r_grad); 
    b.grad <- GRAD (Values.neg r_grad);
    accumulate_gradient b (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let exp_backward r =
  match r.op with
  | EXP a -> begin 
    let r_grad = get_grad r in
    a.grad <- GRAD (Values.mul r.vals r_grad);
    accumulate_gradient a (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let pow_backward r =
  match r.op with
  | POW2 a -> begin
    let r_grad = get_grad r in
    let coef = Values.create (Values.dim a.vals) 2. in
    a.grad <- GRAD (Values.mul (Values.mul coef a.vals) r_grad);
    accumulate_gradient a (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let log_backward r =
  match r.op with
  | LOG a -> begin
    let r_grad = get_grad r in
    a.grad <- GRAD (Values.mul (Values.reciprocal a.vals) r_grad);
    accumulate_gradient a (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let sqrt_backward r = 
  match r.op with
  | SQRT a -> begin
    let r_grad = get_grad r in
    let coef = Values.create (Values.dim a.vals) 0.5 in
    a.grad <- GRAD (Values.mul (Values.mul coef (Values.reciprocal (Values.sqrt a.vals))) r_grad);
    accumulate_gradient a (Values.dim r_grad); 
  end
  | _ -> raise TypeException

let matmul_backward r = 
  match r.op with 
  | MATMUL (a, b) -> begin 
    let r_grad = get_grad r in
    let (n, p), (_, m) = Values.dim a.vals, Values.dim b.vals in 
    a.grad <- GRAD (Values.matmul ~trans_b:112 r_grad b.vals);
    accumulate_gradient a (n, p); 
    b.grad <- GRAD (Values.matmul ~trans_a:112 a.vals r_grad);
    accumulate_gradient b (p, m); 
  end
  | _ -> raise TypeException

let sum_backward r = 
  match r.op with
  | SUM a -> begin
    let r_grad = get_grad r in
    a.grad <- GRAD (Values.create (Values.dim a.vals) r_grad.{0, 0});
    accumulate_gradient a (Values.dim a.vals); 
  end
  | _ -> raise TypeException

let relu_backward r = 
  match r.op with
  | RELU a -> begin
    let r_grad = get_grad r in
    a.grad <- GRAD 
    (Values.mul r_grad 
      (Values.div 
        (Values.add a.vals (Values.abs a.vals)) 
    (Values.mul a.vals (Values.create (Values.dim a.vals) 2.0))));
    accumulate_gradient a (Values.dim a.vals);
  end
  | _ -> raise TypeException

let sigmoid_backward r =
  match r.op with
  | SIGMOID a -> begin 
    let r_grad = get_grad r in
    let dims = Values.dim r.vals in
    a.grad <- GRAD (Values.mul r_grad (Values.mul r.vals (Values.sub (Values.ones dims) r.vals)));
    accumulate_gradient a dims;
  end
  | _ -> raise TypeException


let backward_function_map r = 
  match r.op with
  | ADD (_, _) -> add_backward r;
  | MUL (_, _) -> mul_backward r;
  | SUB (_, _) -> sub_backward r;
  | DIV (_, _) -> div_backward r;
  | NEG _ -> neg_backward r;
  | POW2 _ -> pow_backward r;
  | EXP _ -> exp_backward r;
  | LOG _ -> log_backward r;
  | SQRT _ -> sqrt_backward r;
  | SUM _ -> sum_backward r;
  | MATMUL (_, _) -> matmul_backward r;
  | RELU _ -> relu_backward r;
  | SIGMOID _ -> sigmoid_backward r;
  | CREATE -> ()  

let backward r =
  if (Values.dim r.vals <> (1, 1)) then raise SizeException;
  r.grad <- GRAD (Values.ones (1, 1));
  reverse_topological_sort r backward_function_map
