open Camlgrad.Types
open Camlgrad.Utils

let create (dims: dimensions) (value : float) =
  let tid = random_int () in 
  let vals = Values.create dims value in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = CREATE in
  {tid; vals; grad; acc_grad; op} 

let ones (dims: dimensions) = create dims 1.0
let zeros (dims: dimensions) = create dims 0.0

let from_array (arr : standard_array) = 
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
