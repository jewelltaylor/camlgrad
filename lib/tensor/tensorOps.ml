open Bigarray
open Types 
open Values

let add a b =
  let tid = random_int () in
  let vals = Values.add a.vals b.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = ADD (a, b) in
  {tid; vals; grad; acc_grad; op} 

let sub a b =
  let tid = random_int () in
  let vals = Values.sub a.vals b.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = SUB (a, b) in
  {tid; vals; grad; acc_grad; op} 
  
let mul a b =
  let tid = random_int () in
  let vals = Values.mul a.vals b.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = MUL (a, b) in
  {tid; vals; grad; acc_grad; op} 

let div a b =
  let tid = random_int () in
  let vals = Values.div a.vals b.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = DIV (a, b) in
  {tid; vals; grad; acc_grad; op} 

let matmul a b =
  let tid = random_int () in
  let vals = Values.matmul a.vals b.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = MATMUL (a, b) in
  {tid; vals; grad; acc_grad; op} 

let neg a =
  let tid = random_int () in
  let vals = Values.neg a.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = NEG a in
  {tid; vals; grad; acc_grad; op} 

let exp a =
  let tid = random_int () in
  let vals = Values.exp a.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = EXP a in
  {tid; vals; grad; acc_grad; op} 

let log a =
  let tid = random_int () in
  let vals = Values.log a.vals in
  let grad = NONE in
  let acc_grad = NONE in 
  let op = LOG a in
  {tid; vals; grad; acc_grad; op} 

let sqrt a =
  let tid = random_int () in
  let vals = Values.sqrt a.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = SQRT a in
  {tid; vals; grad; acc_grad; op} 

let pow2 a =
  let tid = random_int () in
  let vals = Values.pow2 a.vals in
  let grad = NONE in 
  let acc_grad = NONE in 
  let op = POW2 a in
  {tid; vals; grad; acc_grad; op} 

let sum a = 
  let tid = random_int () in
  let vals = Values.sum a.vals in
  let grad = NONE in
  let acc_grad = NONE in 
  let op = SUM a in
  {tid; vals; grad; acc_grad; op} 
  
let dim a = (Array2.dim1 a.vals, Array2.dim2 a.vals)

let relu a = 
  let tid = random_int () in
  let dims = Values.dim a.vals in
  let vals = Values.div (Values.add a.vals (Values.abs a.vals)) (Values.create dims 2.0) in
  let grad = NONE in
  let acc_grad = NONE in 
  let op:operator= RELU a in
  {tid; vals; grad; acc_grad; op} 

let sigmoid a = 
  let tid = random_int () in
  let dims = Values.dim a.vals in
  let vals = Values.div (Values.ones dims) ((Values.add (Values.ones dims) (Values.exp (Values.neg a.vals)))) in
  let grad = NONE in
  let acc_grad = NONE in 
  let op:operator = SIGMOID a in
  {tid; vals; grad; acc_grad; op} 
