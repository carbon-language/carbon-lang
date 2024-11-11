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
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+4]]:9: error: package `Core` implicitly referenced here, but not found [CoreNotFound]
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
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+6]]:1: error: function redeclaration differs because return type is `{}` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn K() -> {} { return {}; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn K() -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
fn K() -> {} { return {}; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.2: %.type.2 = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.type.3: type = fn_type @.3 [template]
// CHECK:STDOUT:   %.3: %.type.3 = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = fn_type @I [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.type.4: type = fn_type @.4 [template]
// CHECK:STDOUT:   %.4: %.type.4 = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %.type.5: type = fn_type @.5 [template]
// CHECK:STDOUT:   %.5: %.type.5 = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.type.6: type = fn_type @.6 [template]
// CHECK:STDOUT:   %.7: %.type.6 = struct_value () [template]
// CHECK:STDOUT:   %struct: %.6 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.decl.loc19: %.type.1 = fn_decl @.1 [template = constants.%.1] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_10.2: type = converted %.loc19_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc21_10.2: type = converted %.loc21_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc29: %.type.2 = fn_decl @.2 [template = constants.%.2] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc31_10.2: type = converted %.loc31_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc36: %.type.3 = fn_decl @.3 [template = constants.%.3] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [template = constants.%I] {} {}
// CHECK:STDOUT:   %.decl.loc46: %.type.4 = fn_decl @.4 [template = constants.%.4] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc46_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc46_12.2: type = converted %.loc46_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc48_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc48_12.2: type = converted %.loc48_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc56: %.type.5 = fn_decl @.5 [template = constants.%.5] {} {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc58_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_12.2: type = converted %.loc58_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc65: %.type.6 = fn_decl @.6 [template = constants.%.7] {
// CHECK:STDOUT:     %return.patt: %.6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.6 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc65_12.1: %.6 = struct_literal ()
// CHECK:STDOUT:     %.loc65_12.2: type = converted %.loc65_12.1, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:     %return.param: ref %.6 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.6 = return_slot %return.param
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc46_25: %empty_tuple.type = converted %.loc46_24, %empty_tuple [template = constants.%empty_tuple]
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
// CHECK:STDOUT: fn @.6() -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc65_24: %.6 = struct_literal ()
// CHECK:STDOUT:   %struct: %.6 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc65_25: %.6 = converted %.loc65_24, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc65_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
