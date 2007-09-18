(* RUN: %ocamlc llvm.cma llvm_bitwriter.cma %s -o %t
 * RUN: ./%t %t.bc
 * RUN: llvm-dis < %t.bc > %t.ll
 *)

(* Note: It takes several seconds for ocamlc to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_bitwriter


(* Tiny unit test framework *)
let exit_status = ref 0
let case_num = ref 0

let all_done () =
  prerr_endline "";
  exit !exit_status

let group name =
  prerr_endline "";
  case_num := 0;
  prerr_string ("  " ^ name ^ "... ")

let insist cond =
  incr case_num;
  prerr_char ' ';
  if not cond then begin
     exit_status := 10;
     prerr_char '!'
  end;
  prerr_int !case_num

let suite name f =
  prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module filename


(*===-- Types -------------------------------------------------------------===*)

let test_types () =
  (* RUN: grep {Ty01.*void} < %t.ll
   *)
  group "void";
  insist (add_type_name "Ty01" void_type m);
  insist (Void_type == classify_type void_type);

  (* RUN: grep {Ty02.*i1} < %t.ll
   *)
  group "i1";
  insist (add_type_name "Ty02" i1_type m);
  insist (Integer_type == classify_type i1_type);

  (* RUN: grep {Ty03.*i32} < %t.ll
   *)
  group "i32";
  insist (add_type_name "Ty03" i32_type m);

  (* RUN: grep {Ty04.*i42} < %t.ll
   *)
  group "i42";
  let ty = make_integer_type 42 in
  insist (add_type_name "Ty04" ty m);

  (* RUN: grep {Ty05.*float} < %t.ll
   *)
  group "float";
  insist (add_type_name "Ty05" float_type m);
  insist (Float_type == classify_type float_type);

  (* RUN: grep {Ty06.*double} < %t.ll
   *)
  group "double";
  insist (add_type_name "Ty06" double_type m);
  insist (Double_type == classify_type double_type);

  (* RUN: grep {Ty07.*i32.*i1, double} < %t.ll
   *)
  group "function";
  let ty = make_function_type i32_type [| i1_type; double_type |] false in
  insist (add_type_name "Ty07" ty m);
  insist (Function_type = classify_type ty);
  insist (not (is_var_arg ty));
  insist (i32_type == return_type ty);
  insist (double_type == (param_types ty).(1));
  
  (* RUN: grep {Ty08.*\.\.\.} < %t.ll
   *)
  group "vararg";
  let ty = make_function_type void_type [| i32_type |] true in
  insist (add_type_name "Ty08" ty m);
  insist (is_var_arg ty);
  
  (* RUN: grep {Ty09.*\\\[7 x i8\\\]} < %t.ll
   *)
  group "array";
  let ty = make_array_type i8_type 7 in
  insist (add_type_name "Ty09" ty m);
  insist (7 = array_length ty);
  insist (i8_type == element_type ty);
  insist (Array_type == classify_type ty);
  
  (* RUN: grep {Ty10.*float\*} < %t.ll
   *)
  group "pointer";
  let ty = make_pointer_type float_type in
  insist (add_type_name "Ty10" ty m);
  insist (float_type == element_type ty);
  insist (Pointer_type == classify_type ty);
  
  (* RUN: grep {Ty11.*\<4 x i16\>} < %t.ll
   *)
  group "vector";
  let ty = make_vector_type i16_type 4 in
  insist (add_type_name "Ty11" ty m);
  insist (i16_type == element_type ty);
  insist (4 = vector_size ty);
  
  (* RUN: grep {Ty12.*opaque} < %t.ll
   *)
  group "opaque";
  let ty = make_opaque_type () in
  insist (add_type_name "Ty12" ty m);
  insist (ty == ty);
  insist (ty <> make_opaque_type ())


(*===-- Global Values -----------------------------------------------------===*)

let test_global_values () =
  let (++) x f = f x; x in
  let zero32 = make_null i32_type in

  (* RUN: grep {GVal01} < %t.ll
   *)
  group "naming";
  let g = define_global "TEMPORARY" zero32 m in
  prerr_endline "";
  prerr_endline (value_name g);
  insist ("TEMPORARY" = value_name g);
  set_value_name "GVal01" g;
  insist ("GVal01" = value_name g);

  (* RUN: grep {GVal02.*linkonce} < %t.ll
   *)
  group "linkage";
  let g = define_global "GVal02" zero32 m ++
          set_linkage Link_once_linkage in
  insist (Link_once_linkage = linkage g);

  (* RUN: grep {GVal03.*Hanalei} < %t.ll
   *)
  group "section";
  let g = define_global "GVal03" zero32 m ++
          set_section "Hanalei" in
  insist ("Hanalei" = section g);
  
  (* RUN: grep {GVal04.*hidden} < %t.ll
   *)
  group "visibility";
  let g = define_global "GVal04" zero32 m ++
          set_visibility Hidden_visibility in
  insist (Hidden_visibility = visibility g);
  
  (* RUN: grep {GVal05.*align 128} < %t.ll
   *)
  group "alignment";
  let g = define_global "GVal05" zero32 m ++
          set_alignment 128 in
  insist (128 = alignment g)


(*===-- Global Variables --------------------------------------------------===*)

let test_global_variables () =
  let (++) x f = f x; x in
  let fourty_two32 = make_int_constant i32_type 42 false in

  (* RUN: grep {GVar01.*external} < %t.ll
   *)
  group "declarations";
  let g = declare_global i32_type "GVar01" m in
  insist (is_declaration g);
  
  (* RUN: grep {GVar02.*42} < %t.ll
   * RUN: grep {GVar03.*42} < %t.ll
   *)
  group "definitions";
  let g = define_global "GVar02" fourty_two32 m in
  let g2 = declare_global i32_type "GVar03" m ++
           set_initializer fourty_two32 in
  insist (not (is_declaration g));
  insist (not (is_declaration g2));
  insist ((global_initializer g) == (global_initializer g2));

  (* RUN: grep {GVar04.*thread_local} < %t.ll
   *)
  group "threadlocal";
  let g = define_global "GVar04" fourty_two32 m ++
          set_thread_local true in
  insist (is_thread_local g);

  (* RUN: grep -v {GVar05} < %t.ll
   *)
  let g = define_global "GVar05" fourty_two32 m in
  delete_global g


(*===-- Writer ------------------------------------------------------------===*)

let test_writer () =
  group "writer";
  insist (write_bitcode_file m filename);
  
  dispose_module m


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "types"            test_types;
  suite "global values"    test_global_values;
  suite "global variables" test_global_variables;
  suite "writer"           test_writer;
  all_done ()
