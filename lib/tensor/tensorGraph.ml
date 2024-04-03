open Camlgrad.Types
open TensorUtils

module G = Graph.Imperative.Digraph.ConcreteBidirectional(struct
  type t = string
  let compare = Stdlib.compare
  let hash = Hashtbl.hash
  let equal = (=)
end)

let reverse_topological_sort r f = 
  let rec reverse_topological_sort_helper r =
    match r.op with
    | ADD (a, b) |  MUL (a, b) | SUB (a, b) | DIV (a, b) | MATMUL (a, b) -> begin
      f r;
      reverse_topological_sort_helper a;
      reverse_topological_sort_helper b;
    end
    | NEG a | POW2 a | EXP a | LOG a | SQRT a | SUM a | RELU a | SIGMOID a -> begin
      f r;
      reverse_topological_sort_helper a;
    end
    | CREATE -> ()  
  in reverse_topological_sort_helper r;;

let add_vertices g r =
    match r.op with
    | ADD (a, b) |  MUL (a, b) | SUB (a, b) | DIV (a, b) | MATMUL (a, b) -> begin
      let a_id = tensor_to_vertex_string a in
      let b_id = tensor_to_vertex_string b in
      let op_id = binary_op_to_vertex_string r.op a.tid b.tid in
      let r_id = tensor_to_vertex_string r in

      if not (G.mem_vertex g a_id) then G.add_vertex g a_id; 
      if not (G.mem_vertex g b_id) then G.add_vertex g b_id; 
      if not (G.mem_vertex g op_id) then G.add_vertex g op_id; 
      if not (G.mem_vertex g r_id) then G.add_vertex g r_id; 

      G.add_edge g a_id op_id;
      G.add_edge g b_id op_id;
      G.add_edge g op_id r_id;

    end
    | NEG a | POW2 a | EXP a | LOG a | SQRT a | SUM a | RELU a | SIGMOID a -> begin
      let a_id = tensor_to_vertex_string a in
      let op_id = urnary_op_to_vertex_string r.op a.tid in
      let r_id = tensor_to_vertex_string r in

      if not (G.mem_vertex g a_id) then G.add_vertex g a_id; 
      if not (G.mem_vertex g op_id) then G.add_vertex g op_id; 
      if not (G.mem_vertex g r_id) then G.add_vertex g r_id; 

      G.add_edge g a_id op_id;
      G.add_edge g op_id r_id;
    end
    | CREATE -> ()

let visualize_computation_graph ?(file_name = "graph.dot") r =
  let g = G.create () in
  reverse_topological_sort r (add_vertices g);
  let file = open_out_bin file_name in
  let module Dot = Graph.Graphviz.Dot(struct
    include G
    let graph_attributes _ = []
    let default_vertex_attributes _ = []
    let vertex_name v = "\"" ^ v ^ "\""
    let vertex_attributes _ = []
    let get_subgraph _ = None
    let default_edge_attributes _ = []
    let edge_attributes _ = []
  end) in
  Dot.output_graph file g
