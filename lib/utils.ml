open Types


let range start stop =
 if start >= stop then raise InvalidArgumentException;

 let rec rangeHelper start stop =
   if start == stop then []
   else start :: (rangeHelper (start+1) stop) 
 in rangeHelper start stop

let random_int () = 
  let bound = 10000000 in
  Random.int bound 

