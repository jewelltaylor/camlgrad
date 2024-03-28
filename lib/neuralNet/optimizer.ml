open Camlgrad.Types

let apply_update mlp_layer learning_rate = 
  let weight_grad = mlp_layer.weights.grad in 
  let bias_grad = mlp_layer.bias.grad in
  match (weight_grad, bias_grad) with
  | (GRAD w_grad, GRAD b_grad) -> begin
    let w_per_param_lr = Values.create (Values.dim w_grad) learning_rate in
    let b_per_param_lr = Values.create (Values.dim b_grad) learning_rate in
    mlp_layer.weights.vals <- Values.sub mlp_layer.weights.vals (Values.mul w_per_param_lr w_grad);
    mlp_layer.bias.vals <- Values.sub mlp_layer.bias.vals (Values.mul b_per_param_lr b_grad);
  end
  | _ -> raise TypeException 

let update mlp learning_rate = 
  Array.iter (
    fun mlp_layer -> apply_update mlp_layer learning_rate 
  ) mlp
  
let zero_grad_mlp_layer mlp_layer =
  let weight_grad = mlp_layer.weights.grad in 
  let bias_grad = mlp_layer.bias.grad in
  match (weight_grad, bias_grad) with
  | (GRAD w_grad, GRAD b_grad) -> begin
    mlp_layer.weights.grad <- GRAD (Values.mul w_grad (Values.zeros (Values.dim w_grad)));
    mlp_layer.bias.grad <- GRAD (Values.mul b_grad (Values.zeros (Values.dim b_grad)))
  end
  | _ -> raise TypeException 

let zero_grad mlp =
  Array.iter (
    fun mlp_layer -> zero_grad_mlp_layer mlp_layer
  ) mlp

let stochastic_gradient_descent mlp loss learning_rate =
  Tensor.backward loss;
  update mlp learning_rate
