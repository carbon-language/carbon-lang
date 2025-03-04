// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_call_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_call_in_type.carbon

// TODO: Some variant of this should work, once compile-time function calls are
// supported.
fn ReturnType() -> type { return i32; }
// CHECK:STDERR: fail_call_in_type.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn Six() -> ReturnType() { return 6; }
// CHECK:STDERR:             ^~~~~~~~~~~~
// CHECK:STDERR:
fn Six() -> ReturnType() { return 6; }

// CHECK:STDOUT: --- fail_call_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ReturnType.type: type = fn_type @ReturnType [concrete]
// CHECK:STDOUT:   %ReturnType: %ReturnType.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Six.type: type = fn_type @Six [concrete]
// CHECK:STDOUT:   %Six: %Six.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ReturnType = %ReturnType.decl
// CHECK:STDOUT:     .Six = %Six.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ReturnType.decl: %ReturnType.type = fn_decl @ReturnType [concrete = constants.%ReturnType] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Six.decl: %Six.type = fn_decl @Six [concrete = constants.%Six] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ReturnType.ref: %ReturnType.type = name_ref ReturnType, file.%ReturnType.decl [concrete = constants.%ReturnType]
// CHECK:STDOUT:     %ReturnType.call: init type = call %ReturnType.ref()
// CHECK:STDOUT:     %.loc18_24.1: type = value_of_initializer %ReturnType.call
// CHECK:STDOUT:     %.loc18_24.2: type = converted %ReturnType.call, %.loc18_24.1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnType() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return %i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
