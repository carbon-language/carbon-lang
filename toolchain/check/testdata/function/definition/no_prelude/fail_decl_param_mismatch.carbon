// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_decl_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_decl_param_mismatch.carbon

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
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.9f9306.1: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.57b67a.1: %G.type.9f9306.1 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.9f9306.2: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.57b67a.2: %G.type.9f9306.2 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.6826c6.1: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %H.8a6545.1: %H.type.6826c6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.6826c6.2: type = fn_type @H.2 [concrete]
// CHECK:STDOUT:   %H.8a6545.2: %H.type.6826c6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.af49c4.1: type = fn_type @I.1 [concrete]
// CHECK:STDOUT:   %I.58e4ec.1: %I.type.af49c4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.af49c4.2: type = fn_type @I.2 [concrete]
// CHECK:STDOUT:   %I.58e4ec.2: %I.type.af49c4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %J.type.8e2c84.1: type = fn_type @J.1 [concrete]
// CHECK:STDOUT:   %J.3662e8.1: %J.type.8e2c84.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.8e2c84.2: type = fn_type @J.2 [concrete]
// CHECK:STDOUT:   %J.3662e8.2: %J.type.8e2c84.2 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type.6cc1c5.1: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %K.460d95.1: %K.type.6cc1c5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %K.type.6cc1c5.2: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.460d95.2: %K.type.6cc1c5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc11
// CHECK:STDOUT:     .G = %G.decl.loc21
// CHECK:STDOUT:     .H = %H.decl.loc31
// CHECK:STDOUT:     .I = %I.decl.loc38
// CHECK:STDOUT:     .J = %J.decl.loc48
// CHECK:STDOUT:     .K = %K.decl.loc58
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc11: %F.type.b25846.1 = fn_decl @F.1 [concrete = constants.%F.c41931.1] {} {}
// CHECK:STDOUT:   %F.decl.loc19: %F.type.b25846.2 = fn_decl @F.2 [concrete = constants.%F.c41931.2] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc19_10.1: type = splice_block %.loc19_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc19_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc19_10.3: type = converted %.loc19_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc21: %G.type.9f9306.1 = fn_decl @G.1 [concrete = constants.%G.57b67a.1] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc21_10.1: type = splice_block %.loc21_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc21_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc21_10.3: type = converted %.loc21_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc29: %G.type.9f9306.2 = fn_decl @G.2 [concrete = constants.%G.57b67a.2] {} {}
// CHECK:STDOUT:   %H.decl.loc31: %H.type.6826c6.1 = fn_decl @H.1 [concrete = constants.%H.8a6545.1] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc31_10.1: type = splice_block %.loc31_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc31_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc31_10.3: type = converted %.loc31_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl.loc36: %H.type.6826c6.2 = fn_decl @H.2 [concrete = constants.%H.8a6545.2] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc38: %I.type.af49c4.1 = fn_decl @I.1 [concrete = constants.%I.58e4ec.1] {} {}
// CHECK:STDOUT:   %I.decl.loc46: %I.type.af49c4.2 = fn_decl @I.2 [concrete = constants.%I.58e4ec.2] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc46_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc46_12.2: type = converted %.loc46_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc48: %J.type.8e2c84.1 = fn_decl @J.1 [concrete = constants.%J.3662e8.1] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc48_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc48_12.2: type = converted %.loc48_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc56: %J.type.8e2c84.2 = fn_decl @J.2 [concrete = constants.%J.3662e8.2] {} {}
// CHECK:STDOUT:   %K.decl.loc58: %K.type.6cc1c5.1 = fn_decl @K.1 [concrete = constants.%K.460d95.1] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc58_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_12.2: type = converted %.loc58_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl.loc66: %K.type.6cc1c5.2 = fn_decl @K.2 [concrete = constants.%K.460d95.2] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc66_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc66_12.2: type = converted %.loc66_12.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x.param_patt: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.1(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2(%x.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.2() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc46_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc46_25: %empty_tuple.type = converted %.loc46_24, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc46_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.1() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.1() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc66_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc66_25: %empty_struct_type = converted %.loc66_24, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc66_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
