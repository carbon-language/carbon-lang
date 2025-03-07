// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_type.carbon

fn Mismatch() -> i32 {
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE+7]]:19: error: type `f64` of `returned var` does not match return type of enclosing function [ReturnedVarWrongType]
  // CHECK:STDERR:   returned var v: f64 = 0.0;
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE-4]]:15: note: return type of function is `i32` [ReturnTypeHereNote]
  // CHECK:STDERR: fn Mismatch() -> i32 {
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  returned var v: f64 = 0.0;
  return var;
}

// CHECK:STDOUT: --- fail_returned_var_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Mismatch.type: type = fn_type @Mismatch [concrete]
// CHECK:STDOUT:   %Mismatch: %Mismatch.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mismatch = %Mismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mismatch.decl: %Mismatch.type = fn_decl @Mismatch [concrete = constants.%Mismatch] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mismatch() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: f64 = binding_pattern v
// CHECK:STDOUT:     %.loc19_12: f64 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref f64 = var v
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete = constants.%float]
// CHECK:STDOUT:   assign %v.var, %float
// CHECK:STDOUT:   %.loc19_19.1: type = splice_block %.loc19_19.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc19_19.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc19_19.3: type = converted %float.make_type, %.loc19_19.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref f64 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc19_16: f64 = bind_value %v
// CHECK:STDOUT:   return %.loc19_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
