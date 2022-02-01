(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/transform_utils.ml
 * RUN: %ocamlc -g -w +A -package llvm.transform_utils -linkpkg %t/transform_utils.ml -o %t/executable
 * RUN: %t/executable
 * RUN: %ocamlopt -g -w +A -package llvm.transform_utils -linkpkg %t/transform_utils.ml -o %t/executable
 * RUN: %t/executable
 * XFAIL: vg_leak
 *)

open Llvm
open Llvm_transform_utils

let context = global_context ()

let test_clone_module () =
  let m  = create_module context "mod" in
  let m' = clone_module m in
  if m == m' then failwith "m == m'";
  if string_of_llmodule m <> string_of_llmodule m' then failwith "string_of m <> m'"

let () =
  test_clone_module ()
