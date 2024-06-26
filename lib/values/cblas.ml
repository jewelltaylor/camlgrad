open Ctypes
open Foreign

let cblas_saxpy = 
  foreign "cblas_saxpy" 
    (int @-> float @-> (ptr float) @-> int @-> (ptr float) @-> int @-> returning void)  

let cblas_sdot = 
  foreign "cblas_sdot" 
    (int @-> (ptr float) @-> int @-> (ptr float) @-> int @-> returning float)

let cblas_sgemv = 
  foreign "cblas_sgemv"
    (int @-> int @-> int @-> int @-> float @-> (ptr float) @-> 
      int @-> (ptr float) @-> int @-> float @-> (ptr float) @-> int @-> returning void)

let cblas_sgemm = 
  foreign "cblas_sgemm" 
    (int @-> int @-> int @-> int @-> int @-> int @-> float @-> (ptr float) @-> 
      int @-> (ptr float) @-> int @-> float @-> (ptr float) @-> int @-> returning void)
