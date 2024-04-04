<p align="center">
<img src="https://github.com/VectorInstitute/forecasting-with-dl/assets/34798787/b6e54bca-522d-459d-b498-c5e971acc06d" width="600" height="400" />
</p>
<h1 align="center"> 🐫 camlgrad 📉
</h1>
<p align="center">
<img src="https://github.com/jewelltaylor/camlgrad/actions/workflows/unit-test.yml/badge.svg" />
</p>

Inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad), camlgrad is a toy autograd engine in OCaml from scratch using an Apple Accelerate backend for vectorized computation. **camlgrad** offers a `Tensor` module with a wide variety of unary operations and binary operations that can be composed in arbitrary ways to define computation graphs that admit forward and backward passes on scalar valued functions. Using the `Tensor` module, several other modules are defined for calculating losses (`Loss`), defining multi-layer perceptrons (`Mlp`) and performing gradient descent (`Optimizer`).

## Installation
Install ocaml compiler and package manager (opam) along with some useful platform tools:
```bash
brew install ocaml
opam install ocaml-lsp-server odoc ocamlformat utop
```

Subsequently, we can create a switch (an isolated ocaml environment), activate it and install the packages necessary for **camlgrad**:
```bash
opam switch create camlgrad 5.1
opam switch camlgrad
opam install ocamlgraph ctypes-foreign alcotest
```

## Usage
camlgrad makes it easy to define an MLP and perform forward and backward passes. As illustrated in the code snippet below, we can simply: 
- Define an MLP containing a single layer with sigmoid activation
- Define arbitrary input and target
- Perform a forward pass on the MLP model
- Calculate binary cross entropy
- Update parameters of MLP using gradient descent with respect to the loss

```ocaml
let mlp = Mlp.get_mlp [|(Tensor.sigmoid, (100, 1))|] in
let input = Tensor.random (1, 100) in
let target = Tensor.ones (1, 1) in
let (pred, _) = Mlp.mlp_forward mlp input in 
let bce_loss = Loss.binary_cross_entropy pred target in
Optimizer.gradient_descent mlp bce_loss 0.01;
```

In order to visualize the computation graph, we can export a general specification of the graph into a file using: 
```ocaml
Tensor.visualize_computation_graph bce_loss "graph.dot";
```

To render graph we use the [Graphviz](https://graphviz.org/) CLI:
```bash
dot -Tpng graph.dot -o graph.png
```

<p align="center">
<img src="https://github.com/VectorInstitute/forecasting-with-dl/assets/34798787/5b3430b4-81a5-4eae-af1d-03e4e8a0ae31" width="600" height="800" />
</p>

**Note**: Must first install graphviz with `brew install graphviz`

## Apple Accelerate Details 
[Apple Accelerate](https://developer.apple.com/documentation/accelerate) is a set of API's to perform large-scale mathematical computations and image calculations, optimized for high performance and low energy consumption. The 2 APIs I leverage in particular are: 
- [BLAS](https://developer.apple.com/documentation/accelerate/blas): Perform common linear algebra operations with Apple’s implementation of the Basic Linear Algebra Subprograms (BLAS).
- [VForce](https://developer.apple.com/documentation/accelerate/veclib/vforce): Perform transcendental and trigonometric functions on vectors of any length.

The BLAS and VForce API are in C which we can easily interface with from OCaml using the [Ctypes library](https://github.com/yallop/ocaml-ctypes). The Ctypes library lets you define the C interface in pure OCaml, and the library then takes care of loading the C symbols and invoking the foreign function call.  

To represent tensors, the [OCaml Bigarray](https://v2.ocaml.org/api/Bigarray.html) is used. Bigrarray implements multi-dimensional arrays of integers and floating-point numbers. In particular, it allows efficient sharing of large numerical arrays between OCaml code and C or Fortran numerical libraries.

**camlgrad** only requires a few libraries outside the standard library ([alcotest](https://github.com/mirage/alcotest) for testing, [ctypes-foreign](https://github.com/yallop/ocaml-ctypes) for foreign function interface and [ocamlgraph](https://github.com/backtracking/ocamlgraph) for generating viusalization of computation graph). 

## Contributing
In the unlikely event someone read this far and is interested in contributing, feel free to put up an issue or pull request 😊 <p align="center">

