open Bigarray
open Types 
open Utils

let add a b =
  let tid = random_int in
  let vals = Values.add a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = ADD (a, b) in
  {tid; vals; grad; op}

let sub a b =
  let tid = random_int in
  let vals = Values.sub a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = SUB (a, b) in
  {tid; vals; grad; op}
  
let mul a b =
  let tid = random_int in
  let vals = Values.mul a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = MUL (a, b) in
  {tid; vals; grad; op}

let div a b =
  let tid = random_int in
  let vals = Values.div a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = DIV (a, b) in
  {tid; vals; grad; op}

let matmul a b =
  let tid = random_int in
  let vals = Values.matmul a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = MATMUL (a, b) in
  {tid; vals; grad; op}

let matvecmul a b =
  let tid = random_int in
  let vals = Values.matvecmul a.vals b.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = MATVECMUL (a, b) in
  {tid; vals; grad; op}

let neg a =
  let tid = random_int in
  let vals = Values.neg a.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = NEG a in
  {tid; vals; grad; op}

let exp a =
  let tid = random_int in
  let vals = Values.exp a.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = EXP a in
  {tid; vals; grad; op}

let log a =
  let tid = random_int in
  let vals = Values.log a.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = LOG a in
  {tid; vals; grad; op}

let sqrt a =
  let tid = random_int in
  let vals = Values.sqrt a.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = SQRT a in
  {tid; vals; grad; op}

let pow2 a =
  let tid = random_int in
  let vals = Values.pow2 a.vals in
  let grad = Values.zeros (Values.dims vals) in
  let op = POW2 a in
  {tid; vals; grad; op}

let dim a = 
  match a.vals with
  | V1D a1D -> D1D (Array1.dim a1D)
  | V2D a2D -> D2D (Array2.dim1 a2D, Array2.dim2 a2D)
