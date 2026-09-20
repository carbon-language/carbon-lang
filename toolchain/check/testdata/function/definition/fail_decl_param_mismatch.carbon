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
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-20: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-7: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR:
fn F(unused x: ()) {}

fn G(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-8: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-12: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR:
fn G() {}

fn H(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+2]]:16-19: error: `Core.Bool` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR:
fn H(unused x: bool) {}

fn I();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-14: error: function redeclaration differs because return type is `()` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-7: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
// CHECK:STDERR:
fn I() -> () { return (); }

fn J() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-8: error: function redeclaration differs because no return type is provided [FunctionRedeclReturnTypeDiffersNoReturn]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-13: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:
fn J() {}

fn K() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-14: error: function redeclaration differs because return type is `{}` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-13: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:
fn K() -> {} { return {}; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.eb3ec9.1: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.7088cc.1: %F.type.eb3ec9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.2180a6.1: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type.eb3ec9.2: type = fn_type @F.loc19 [concrete]
// CHECK:STDOUT:   %F.7088cc.2: %F.type.eb3ec9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt.2180a6.2: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type.f42045.1: type = fn_type @G.loc21 [concrete]
// CHECK:STDOUT:   %G.4228ed.1: %G.type.f42045.1 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.f42045.2: type = fn_type @G.loc25 [concrete]
// CHECK:STDOUT:   %G.4228ed.2: %G.type.f42045.2 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.c0bf33.1: type = fn_type @H.loc27 [concrete]
// CHECK:STDOUT:   %H.aacb46.1: %H.type.c0bf33.1 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.c0bf33.2: type = fn_type @H.loc30 [concrete]
// CHECK:STDOUT:   %H.aacb46.2: %H.type.c0bf33.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.4c8936.1: type = fn_type @I.loc32 [concrete]
// CHECK:STDOUT:   %I.8a67f5.1: %I.type.4c8936.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %I.type.4c8936.2: type = fn_type @I.loc36 [concrete]
// CHECK:STDOUT:   %I.8a67f5.2: %I.type.4c8936.2 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.6304f1.1: type = fn_type @J.loc38 [concrete]
// CHECK:STDOUT:   %J.64b370.1: %J.type.6304f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.6304f1.2: type = fn_type @J.loc42 [concrete]
// CHECK:STDOUT:   %J.64b370.2: %J.type.6304f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type.b73f4f.1: type = fn_type @K.loc44 [concrete]
// CHECK:STDOUT:   %K.4fd1e7.1: %K.type.b73f4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.469: Core.Form = init_form %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %return.param_patt.296: %pattern_type.a96 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c45: %pattern_type.a96 = return_slot_pattern %return.param_patt.296, %empty_struct_type [concrete]
// CHECK:STDOUT:   %K.type.b73f4f.2: type = fn_type @K.loc48 [concrete]
// CHECK:STDOUT:   %K.4fd1e7.2: %K.type.b73f4f.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc15
// CHECK:STDOUT:     .G = %G.decl.loc21
// CHECK:STDOUT:     .H = %H.decl.loc27
// CHECK:STDOUT:     .I = %I.decl.loc32
// CHECK:STDOUT:     .J = %J.decl.loc38
// CHECK:STDOUT:     .K = %K.decl.loc44
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc15: %F.type.eb3ec9.1 = fn_decl @F.loc15 [concrete = constants.%F.7088cc.1] {} {}
// CHECK:STDOUT:   %F.decl.loc19: %F.type.eb3ec9.2 = fn_decl @F.loc19 [concrete = constants.%F.7088cc.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc19_17.1: type = splice_block %.loc19_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc19_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc19_17.3: type = converted %.loc19_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc21: %G.type.f42045.1 = fn_decl @G.loc21 [concrete = constants.%G.4228ed.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc21_10.1: type = splice_block %.loc21_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc21_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc21_10.3: type = converted %.loc21_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc25: %G.type.f42045.2 = fn_decl @G.loc25 [concrete = constants.%G.4228ed.2] {} {}
// CHECK:STDOUT:   %H.decl.loc27: %H.type.c0bf33.1 = fn_decl @H.loc27 [concrete = constants.%H.aacb46.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc27_10.1: type = splice_block %.loc27_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc27_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc27_10.3: type = converted %.loc27_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl.loc30: %H.type.c0bf33.2 = fn_decl @H.loc30 [concrete = constants.%H.aacb46.2] {
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %x.patt: <error> = wrapper_binding_pattern x, %x.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc30: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:     %x: <error> = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc32: %I.type.4c8936.1 = fn_decl @I.loc32 [concrete = constants.%I.8a67f5.1] {} {}
// CHECK:STDOUT:   %I.decl.loc36: %I.type.4c8936.2 = fn_decl @I.loc36 [concrete = constants.%I.8a67f5.2] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc36_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc36_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc36_12.2: type = converted %.loc36_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc36_12.3: Core.Form = init_form %.loc36_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc38: %J.type.6304f1.1 = fn_decl @J.loc38 [concrete = constants.%J.64b370.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc38_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc38_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc38_12.2: type = converted %.loc38_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc38_12.3: Core.Form = init_form %.loc38_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc42: %J.type.6304f1.2 = fn_decl @J.loc42 [concrete = constants.%J.64b370.2] {} {}
// CHECK:STDOUT:   %K.decl.loc44: %K.type.b73f4f.1 = fn_decl @K.loc44 [concrete = constants.%K.4fd1e7.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc44_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc44_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc44_12.2: type = converted %.loc44_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc44_12.3: Core.Form = init_form %.loc44_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl.loc48: %K.type.b73f4f.2 = fn_decl @K.loc48 [concrete = constants.%K.4fd1e7.2] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern [concrete = constants.%return.param_patt.296]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern %return.param_patt, %.loc48_12.2 [concrete = constants.%return.patt.c45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc48_12.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc48_12.2: type = converted %.loc48_12.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc48_12.3: Core.Form = init_form %.loc48_12.2 [concrete = constants.%.469]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc15();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc19(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc21(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc25() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc27(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc30(%x.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc32();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc36() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc36_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36_24.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36_25: init %empty_tuple.type = converted %.loc36_24.1, %.loc36_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc36_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc38() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc42() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc44() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc48() -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc48_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc48_24.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc48_25: init %empty_struct_type = converted %.loc48_24.1, %.loc48_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc48_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
