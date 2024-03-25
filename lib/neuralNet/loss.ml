
let mean_squared_error pred target =
  let (d1, d2) = Tensor.dim pred in
  let error = Tensor.sub pred target in
  let squared_error = Tensor.pow2 error in
  let total_squared_error = Tensor.sum squared_error in
  Tensor.div total_squared_error (Tensor.create (1, 1) (float_of_int (d1*d2)))

let binary_cross_entropy pred target =
  let (d1, d2) = Tensor.dim pred in
  let pos_class_loss = Tensor.mul target (Tensor.log pred) in
  let neg_class_loss = Tensor.mul (Tensor.sub (Tensor.ones (d1, d2)) target) (Tensor.log (Tensor.sub (Tensor.ones (d1, d2)) pred)) in
  let total_loss = Tensor.sum (Tensor.add pos_class_loss neg_class_loss) in
  Tensor.neg (Tensor.div total_loss (Tensor.create (1, 1) (float_of_int (d1*d2))))
