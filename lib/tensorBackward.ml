open Types


let add_backward r = 
  match r.op with
  | ADD (a, b) -> begin 
    a.grad <- Values.add a.grad r.grad;
    b.grad <- Values.add b.grad r.grad;
  end
  | _ -> raise TypeException

let mul_backward r = 
  match r.op with
  | MUL (a, b) -> begin 
    a.grad <- Values.add a.grad (Values.mul b.vals r.grad);
    b.grad <- Values.add b.grad (Values.mul a.vals r.grad);
  end
  | _ -> raise TypeException

let div_backward r = 
  match r.op with
  | DIV (a, b) -> begin 
    a.grad <- Values.add a.grad (Values.div b.vals r.grad);
    b.grad <- Values.add b.grad (Values.div a.vals r.grad);
  end
  | _ -> raise TypeException

let neg_backward r = 
  match r.op with
  | NEG a -> begin 
    let b = Values.create (Values.dim a.vals) (-1.0) in 
    a.grad <- Values.add a.grad (Values.mul b r.grad);
  end
  | _ -> raise TypeException

let exp_backward r =
  match r.op with
  | EXP a -> a.grad <- Values.add a.grad (Values.mul r.vals r.grad)
  | _ -> raise TypeException

let pow_backward r =
  match r.op with
  | POW2 a -> begin
    let coef = Values.create (Values.dim a.vals) 2. in
    a.grad <- Values.add a.grad (Values.mul (Values.mul coef a.vals) r.grad)
  end
  | _ -> raise TypeException

let log_backward r =
  match r.op with
  | LOG a -> begin 
      a.grad <- Values.add a.grad (Values.mul (Values.reciprocal a.vals) r.grad)
  end
  | _ -> raise TypeException

let sqrt_backward r = 
  match r.op with
  | SQRT a -> begin
    let coef = Values.create (Values.dim a.vals) 0.5 in
    a.grad <- Values.add a.grad (Values.mul (Values.mul coef (Values.reciprocal r.vals)) r.grad)
  end
  | _ -> raise TypeException

let backward r = 
  r.grad <- Values.ones (Values.dim r.grad);
  match r.op with
  | ADD (_, _) -> add_backward r
  | MUL (_, _) -> mul_backward r
  | DIV (_, _) -> div_backward r
  | NEG (_) -> neg_backward r
  | POW2 (_) -> pow_backward r
  | EXP (_) -> exp_backward r
  | LOG (_) -> log_backward r
  | SQRT (_) -> sqrt_backward r
  | _ -> raise TypeException
