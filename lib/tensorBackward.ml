open Types
open TensorUtils


let add_backward r = 
  match r.op with
  | ADD (a, b) -> begin 
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad, b_grad = get_grad a.grad dims, get_grad b.grad dims in 
    a.grad <- GRAD (Values.add a_grad r_grad); 
    b.grad <- GRAD (Values.add b_grad r_grad);
  end
  | _ -> raise TypeException

let mul_backward r = 
  match r.op with
  | MUL (a, b) -> begin
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad, b_grad = get_grad a.grad dims, get_grad b.grad dims in 
    a.grad <- GRAD (Values.add a_grad (Values.mul b.vals r_grad)); 
    b.grad <- GRAD (Values.add b_grad (Values.mul a.vals r_grad));
  end
  | _ -> raise TypeException

let div_backward r = 
  match r.op with
  | DIV (a, b) -> begin 
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad, b_grad = get_grad a.grad dims, get_grad b.grad dims in 
    a.grad <- GRAD (Values.add a_grad (Values.div b.vals r_grad));
    b.grad <- GRAD (Values.add b_grad (Values.div a.vals r_grad));
  end
  | _ -> raise TypeException

let neg_backward r = 
  match r.op with
  | NEG a -> begin 
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad = get_grad a.grad dims in
    let b = Values.create (Values.dim a.vals) (-1.0) in 
    a.grad <- GRAD (Values.add a_grad (Values.mul b r_grad));
  end
  | _ -> raise TypeException

let exp_backward r =
  match r.op with
  | EXP a -> begin 
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad = get_grad a.grad dims in
    a.grad <- GRAD (Values.add a_grad (Values.mul r.vals r_grad));
  end
  | _ -> raise TypeException

let pow_backward r =
  match r.op with
  | POW2 a -> begin
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad = get_grad a.grad dims in
    let coef = Values.create (Values.dim a.vals) 2. in
    a.grad <- GRAD (Values.add a_grad (Values.mul (Values.mul coef a.vals) r_grad));
  end
  | _ -> raise TypeException

let log_backward r =
  match r.op with
  | LOG a -> begin
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad = get_grad a.grad dims in
    a.grad <- GRAD (Values.add a_grad (Values.mul (Values.reciprocal a.vals) r_grad));
  end
  | _ -> raise TypeException

let sqrt_backward r = 
  match r.op with
  | SQRT a -> begin
    let r_grad = get_r_grad r in
    let dims = Values.dim r_grad in
    let a_grad = get_grad a.grad dims in
    let coef = Values.create (Values.dim a.vals) 0.5 in
    a.grad <- GRAD (Values.add a_grad (Values.mul (Values.mul coef (Values.reciprocal r.vals)) r_grad));
  end
  | _ -> raise TypeException

let matmul_backward r = 
  match r.op with 
  | MATMUL (a, b) -> begin 
    let r_grad = get_r_grad r in
    let (n, p), (_, m) = Values.dim a.vals, Values.dim b.vals in 
    let a_grad = Values.create (n, p) 0.0 in
    let b_grad = Values.create (p, m) 0.0 in
    a.grad <- GRAD (Values.add a_grad (Values.matmul ~trans_b:112 r_grad b.vals));
    b.grad <- GRAD (Values.add b_grad (Values.matmul ~trans_a:112 a.vals r_grad));
  end
  | _ -> raise TypeException

let backward r = 
  let r_grad = get_r_grad r in
  r.grad <- GRAD (Values.ones (Values.dim r_grad));
  match r.op with
  | ADD (_, _) -> add_backward r
  | MUL (_, _) -> mul_backward r
  | DIV (_, _) -> div_backward r
  | NEG (_) -> neg_backward r
  | POW2 (_) -> pow_backward r
  | EXP (_) -> exp_backward r
  | LOG (_) -> log_backward r
  | SQRT (_) -> sqrt_backward r
  | MATMUL (_, _) -> matmul_backward r
  | _ -> raise TypeException
