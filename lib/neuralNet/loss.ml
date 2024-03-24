
let mean_squared_error pred target =
  let (d1, d2) = Tensor.dim pred in
  let error = Tensor.sub pred target in
  let squared_error = Tensor.pow2 error in
  let total_squared_error = Tensor.sum squared_error in
  Tensor.div total_squared_error (Tensor.create (1, 1) (float_of_int (d1*d2)))



