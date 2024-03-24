open Ctypes
open Foreign

let vvexpf = 
  foreign "vvexpf" 
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvlogf = 
  foreign "vvlogf" 
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvsqrtf =
  foreign "vvsqrtf" 
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvpowf =
  foreign "vvpowf"
    ((ptr float) @-> (ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvdivf =
  foreign "vvdivf" 
    ((ptr float) @-> (ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvrecf =
  foreign "vvrecf"
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)

let vvfabsf =
  foreign "vvfabsf"
    ((ptr float) @-> (ptr float) @-> (ptr int) @-> returning void)
