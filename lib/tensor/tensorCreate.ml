open Types
open TensorUtils

let create dims value =
  let tid = random_int () in 
  let vals = Values.create dims value in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = CREATE in
  {tid; vals; grad; acc_grad; op} 

let ones dims = create dims 1.0
let zeros dims = create dims 0.0

let from_array arr = 
  let tid = random_int () in
  let vals = Values.from_array arr in 
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = CREATE in
  {tid; vals; grad; acc_grad; op} 

let random dims = 
  let tid = random_int () in
  let vals = Values.random dims in 
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = CREATE in
  {tid; vals; grad; acc_grad; op} 
