open Ctypes
open Foreign

let cblas_saxpy = 
  foreign "cblas_saxpy" 
    (int @-> float @-> (ptr float) @-> int @-> (ptr float) @-> int @-> returning void)  


