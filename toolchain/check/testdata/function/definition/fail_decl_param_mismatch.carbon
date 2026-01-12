// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/fail_decl_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/fail_decl_param_mismatch.carbon

fn F();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fn F(x: ()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn F(x: ()) {}

fn G(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fn G() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: fn G(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn G() {}

fn H(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+4]]:9: error: `Core.Bool` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn H(x: bool) {}
// CHECK:STDERR:         ^~~~
// CHECK:STDERR:
fn H(x: bool) {}

fn I();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `()` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn I() -> () { return (); }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
// CHECK:STDERR: fn I();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn I() -> () { return (); }

fn J() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: error: function redeclaration differs because no return type is provided [FunctionRedeclReturnTypeDiffersNoReturn]
// CHECK:STDERR: fn J() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn J() -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn J() {}

fn K() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `{}` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn K() -> {} { return {}; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn K() -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn K() -> {} { return {}; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc23 [concrete]
// CHECK:STDOUT:   %F.d98bd5.2: %F.type.117dbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.90f93f.1: type = fn_type @G.loc25 [concrete]
// CHECK:STDOUT:   %G.4dbc02.1: %G.type.90f93f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.90f93f.2: type = fn_type @G.loc33 [concrete]
// CHECK:STDOUT:   %G.4dbc02.2: %G.type.90f93f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.7917a6.1: type = fn_type @H.loc35 [concrete]
// CHECK:STDOUT:   %H.414c03.1: %H.type.7917a6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.7917a6.2: type = fn_type @H.loc40 [concrete]
// CHECK:STDOUT:   %H.414c03.2: %H.type.7917a6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f30c8e.1: type = fn_type @I.loc42 [concrete]
// CHECK:STDOUT:   %I.73832e.1: %I.type.f30c8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.62c: form = init_form %empty_tuple.type, call_param0 [concrete]
// CHECK:STDOUT:   %I.type.f30c8e.2: type = fn_type @I.loc50 [concrete]
// CHECK:STDOUT:   %I.73832e.2: %I.type.f30c8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.fe0423.1: type = fn_type @J.loc52 [concrete]
// CHECK:STDOUT:   %J.6d8208.1: %J.type.fe0423.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.fe0423.2: type = fn_type @J.loc60 [concrete]
// CHECK:STDOUT:   %J.6d8208.2: %J.type.fe0423.2 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type.4b1c50.1: type = fn_type @K.loc62 [concrete]
// CHECK:STDOUT:   %K.548a8d.1: %K.type.4b1c50.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.e6e: form = init_form %empty_struct_type, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %K.type.4b1c50.2: type = fn_type @K.loc70 [concrete]
// CHECK:STDOUT:   %K.548a8d.2: %K.type.4b1c50.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc15
// CHECK:STDOUT:     .G = %G.decl.loc25
// CHECK:STDOUT:     .H = %H.decl.loc35
// CHECK:STDOUT:     .I = %I.decl.loc42
// CHECK:STDOUT:     .J = %J.decl.loc52
// CHECK:STDOUT:     .K = %K.decl.loc62
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc15: %F.type.117dbc.1 = fn_decl @F.loc15 [concrete = constants.%F.d98bd5.1] {} {}
// CHECK:STDOUT:   %F.decl.loc23: %F.type.117dbc.2 = fn_decl @F.loc23 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc23_10.1: type = splice_block %.loc23_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc23_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc23_10.3: type = converted %.loc23_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc25: %G.type.90f93f.1 = fn_decl @G.loc25 [concrete = constants.%G.4dbc02.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc25_10.1: type = splice_block %.loc25_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc25_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc25_10.3: type = converted %.loc25_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc33: %G.type.90f93f.2 = fn_decl @G.loc33 [concrete = constants.%G.4dbc02.2] {} {}
// CHECK:STDOUT:   %H.decl.loc35: %H.type.7917a6.1 = fn_decl @H.loc35 [concrete = constants.%H.414c03.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc35_10.1: type = splice_block %.loc35_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc35_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc35_10.3: type = converted %.loc35_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl.loc40: %H.type.7917a6.2 = fn_decl @H.loc40 [concrete = constants.%H.414c03.2] {
// CHECK:STDOUT:     %x.patt: <error> = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %x: <error> = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc42: %I.type.f30c8e.1 = fn_decl @I.loc42 [concrete = constants.%I.73832e.1] {} {}
// CHECK:STDOUT:   %I.decl.loc50: %I.type.f30c8e.2 = fn_decl @I.loc50 [concrete = constants.%I.73832e.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc50_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc50_12.2: type = converted %.loc50_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc50_12.3: form = init_form %.loc50_12.2, call_param0 [concrete = constants.%.62c]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc52: %J.type.fe0423.1 = fn_decl @J.loc52 [concrete = constants.%J.6d8208.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc52_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc52_12.2: type = converted %.loc52_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc52_12.3: form = init_form %.loc52_12.2, call_param0 [concrete = constants.%.62c]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc60: %J.type.fe0423.2 = fn_decl @J.loc60 [concrete = constants.%J.6d8208.2] {} {}
// CHECK:STDOUT:   %K.decl.loc62: %K.type.4b1c50.1 = fn_decl @K.loc62 [concrete = constants.%K.548a8d.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc62_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc62_12.2: type = converted %.loc62_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc62_12.3: form = init_form %.loc62_12.2, call_param0 [concrete = constants.%.62c]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl.loc70: %K.type.4b1c50.2 = fn_decl @K.loc70 [concrete = constants.%K.548a8d.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc70_12.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc70_12.2: type = converted %.loc70_12.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc70_12.3: form = init_form %.loc70_12.2, call_param0 [concrete = constants.%.e6e]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc15();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc23(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc25(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc33() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc35(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc40(%x.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc42();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc50() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc50_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc50_24.2: init %empty_tuple.type = tuple_init () to %return.param [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc50_25: init %empty_tuple.type = converted %.loc50_24.1, %.loc50_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc50_25 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc52() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc60() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc62() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc70() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc70_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc70_24.2: init %empty_struct_type = struct_init () to %return.param [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc70_25: init %empty_struct_type = converted %.loc70_24.1, %.loc70_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc70_25 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
