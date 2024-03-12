open Types
open Utils


let create (dims: dimensions) (value : float) =
  let tid = random_int in 
  let vals = Values.create dims value in
  let grad = Values.zeros dims in 
  let op = Types.CREATE in
  {tid; vals; grad; op} 

let ones (dims: dimensions) = create dims 1.0
let zeros (dims: dimensions) = create dims 1.0

let from_array (arr : standard_array) = 
  let tid = random_int in
  let vals = Values.from_array arr in 
  let grad = Values.zeros (Values.dims vals) in
  let op = Types.CREATE in
  {tid; vals; grad; op}
