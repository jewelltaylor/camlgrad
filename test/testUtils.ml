open Alcotest

exception SizeException
exception InvalidArgumentException

let range start stop =
 if start >= stop then raise InvalidArgumentException;

 let rec rangeHelper start stop =
   if start == stop then []
   else start :: (rangeHelper (start+1) stop) 
 in rangeHelper start stop

let check_equal a b =
  if (Values.dim a <> Values.dim b) then raise SizeException;
  let (d1, d2) = Values.dim a in
  List.iter (fun i -> 
    List.iter (fun j -> 
      let epsilon_float = 0.0001 in 
      check (float epsilon_float) "Equal Float" a.{i, j} b.{i, j}
  )(range 0 d2)) (range 0 d1)
