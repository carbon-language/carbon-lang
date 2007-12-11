(* RUN: %ocamlc llvm.cma llvm_bitreader.cma llvm_bitwriter.cma %s -o %t
 * RUN: ./%t %t.bc
 * RUN: llvm-dis < %t.bc | grep caml_int_ty
 *)

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let test x = if not x then exit 1 else ()

let _ =
  let fn = Sys.argv.(1) in
  let m = Llvm.create_module "ocaml_test_module" in
  
  ignore (Llvm.define_type_name "caml_int_ty" Llvm.i32_type m);
  
  test (Llvm_bitwriter.write_bitcode_file m fn);
  
  Llvm.dispose_module m;
  
  test (match Llvm_bitreader.read_bitcode_file fn with
  | Llvm_bitreader.Bitreader_success m -> Llvm.dispose_module m; true
  | Llvm_bitreader.Bitreader_failure _ -> false)
