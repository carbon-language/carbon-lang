(* RUN: %ocamlc -warn-error A llvm.cma llvm_executionengine.cma %s -o %t
 * RUN: ./%t %t.bc
 *)

open Llvm
open Llvm_executionengine

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let bomb msg =
  prerr_endline msg;
  exit 2

let define_main_fn m retval =
  let fn =
    let str_arr_type = pointer_type (pointer_type i8_type) in
    define_function "main" (function_type i32_type [| i32_type;
                                                      str_arr_type;
                                                      str_arr_type |]) m in
  let b = builder_at_end (entry_block fn) in
  ignore (build_ret (const_int i32_type retval) b);
  fn

let define_plus m =
  let fn = define_function "plus" (function_type i32_type [| i32_type;
                                                             i32_type |]) m in
  let b = builder_at_end (entry_block fn) in
  let add = build_add (param fn 0) (param fn 1) "sum" b in
  ignore (build_ret add b)

let test_genericvalue () =
  let tu = (1, 2) in
  let ptrgv = GenericValue.of_pointer tu in
  assert (tu = GenericValue.as_pointer ptrgv);
  
  let fpgv = GenericValue.of_float double_type 2. in
  assert (2. = GenericValue.as_float double_type fpgv);
  
  let intgv = GenericValue.of_int i32_type 3 in
  assert (3  = GenericValue.as_int intgv);
  
  let i32gv = GenericValue.of_int32 i32_type 4l in
  assert (4l = GenericValue.as_int32 i32gv);
  
  let nigv = GenericValue.of_nativeint i32_type 5n in
  assert (5n = GenericValue.as_nativeint nigv);
  
  let i64gv = GenericValue.of_int64 i64_type 6L in
  assert (6L = GenericValue.as_int64 i64gv)

let test_executionengine () =
  (* create *)
  let m = create_module "test_module" in
  let main = define_main_fn m 42 in
  
  let m2 = create_module "test_module2" in
  define_plus m2;
  
  let ee = ExecutionEngine.create (ModuleProvider.create m) in
  let mp2 = ModuleProvider.create m2 in
  ExecutionEngine.add_module_provider mp2 ee;
  
  (* run_static_ctors *)
  ExecutionEngine.run_static_ctors ee;
  
  (* run_function_as_main *)
  let res = ExecutionEngine.run_function_as_main main [|"test"|] [||] ee in
  if 42 != res then bomb "main did not return 42";
  
  (* free_machine_code *)
  ExecutionEngine.free_machine_code main ee;
  
  (* find_function *)
  match ExecutionEngine.find_function "dne" ee with
  | Some _ -> raise (Failure "find_function 'dne' failed")
  | None ->
  
  match ExecutionEngine.find_function "plus" ee with
  | None -> raise (Failure "find_function 'plus' failed")
  | Some plus ->
  
  (* run_function *)
  let res = ExecutionEngine.run_function plus
                                         [| GenericValue.of_int i32_type 2;
                                            GenericValue.of_int i32_type 2 |]
                                         ee in
  if 4 != GenericValue.as_int res then bomb "plus did not work";
  
  (* remove_module_provider *)
  Llvm.dispose_module (ExecutionEngine.remove_module_provider mp2 ee);
  
  (* run_static_dtors *)
  ExecutionEngine.run_static_dtors ee;
  
  (* dispose *)
  ExecutionEngine.dispose ee

let _ =
  test_genericvalue ();
  test_executionengine ()
