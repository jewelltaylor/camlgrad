open Ctypes
open Foreign

let vvexpf = 
  foreign "vvexpf" 
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvlogf = 
  foreign "vvlogf" 
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)
