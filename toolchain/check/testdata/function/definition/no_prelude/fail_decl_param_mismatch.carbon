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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b6a92a.1: type = fn_type @.1 [concrete]
// CHECK:STDOUT:   %.d852be.1: %.type.b6a92a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b6a92a.2: type = fn_type @.2 [concrete]
// CHECK:STDOUT:   %.d852be.2: %.type.b6a92a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b6a92a.3: type = fn_type @.3 [concrete]
// CHECK:STDOUT:   %.d852be.3: %.type.b6a92a.3 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b6a92a.4: type = fn_type @.4 [concrete]
// CHECK:STDOUT:   %.d852be.4: %.type.b6a92a.4 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type.b6a92a.5: type = fn_type @.5 [concrete]
// CHECK:STDOUT:   %.d852be.5: %.type.b6a92a.5 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.type.b6a92a.6: type = fn_type @.6 [concrete]
// CHECK:STDOUT:   %.d852be.6: %.type.b6a92a.6 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %.decl.loc19: %.type.b6a92a.1 = fn_decl @.1 [concrete = constants.%.d852be.1] {
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
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT:   %.decl.loc29: %.type.b6a92a.2 = fn_decl @.2 [concrete = constants.%.d852be.2] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
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
// CHECK:STDOUT:   %.decl.loc36: %.type.b6a92a.3 = fn_decl @.3 [concrete = constants.%.d852be.3] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {} {}
// CHECK:STDOUT:   %.decl.loc46: %.type.b6a92a.4 = fn_decl @.4 [concrete = constants.%.d852be.4] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc46_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc46_12.2: type = converted %.loc46_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc48_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc48_12.2: type = converted %.loc48_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc56: %.type.b6a92a.5 = fn_decl @.5 [concrete = constants.%.d852be.5] {} {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [concrete = constants.%K] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc58_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_12.2: type = converted %.loc58_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc66: %.type.b6a92a.6 = fn_decl @.6 [concrete = constants.%.d852be.6] {
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
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%x.param_patt: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.3(%x.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.4() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc46_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc46_25: %empty_tuple.type = converted %.loc46_24, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc46_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.6() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc66_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc66_25: %empty_struct_type = converted %.loc66_24, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc66_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
