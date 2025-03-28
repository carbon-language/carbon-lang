// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_let_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_let_in_type.carbon

let x: type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn Six() -> x { return 6; }

// TODO: This should probably work.
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:5: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
// CHECK:STDERR: let y:! type = i32;
// CHECK:STDERR:     ^~~~~~~~
// CHECK:STDERR:
let y:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:19: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                   ^
// CHECK:STDERR:
fn HalfDozen() -> y { return 6; }

// TODO: This should work.
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:5: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
// CHECK:STDERR: let template z:! type = i32;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let template z:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:28: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn FirstPerfectNumber() -> z { return 6; }
// CHECK:STDERR:                            ^
// CHECK:STDERR:
fn FirstPerfectNumber() -> z { return 6; }

// CHECK:STDOUT: --- fail_let_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Six.type: type = fn_type @Six [concrete]
// CHECK:STDOUT:   %Six: %Six.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %HalfDozen.type: type = fn_type @HalfDozen [concrete]
// CHECK:STDOUT:   %HalfDozen: %HalfDozen.type = struct_value () [concrete]
// CHECK:STDOUT:   %FirstPerfectNumber.type: type = fn_type @FirstPerfectNumber [concrete]
// CHECK:STDOUT:   %FirstPerfectNumber: %FirstPerfectNumber.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Six = %Six.decl
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .HalfDozen = %HalfDozen.decl
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:     .FirstPerfectNumber = %FirstPerfectNumber.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: type = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: type = bind_name x, @__global_init.%i32.loc11
// CHECK:STDOUT:   %Six.decl: %Six.type = fn_decl @Six [concrete = constants.%Six] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.ref: type = name_ref x, file.%x
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: type = binding_pattern y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: type = bind_name y, @__global_init.%i32.loc23
// CHECK:STDOUT:   %HalfDozen.decl: %HalfDozen.type = fn_decl @HalfDozen [concrete = constants.%HalfDozen] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %y.ref: type = name_ref y, file.%y
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: type = binding_pattern z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z: type = bind_name z, @__global_init.%i32.loc35
// CHECK:STDOUT:   %FirstPerfectNumber.decl: %FirstPerfectNumber.type = fn_decl @FirstPerfectNumber [concrete = constants.%FirstPerfectNumber] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %z.ref: type = name_ref z, file.%z
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HalfDozen() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FirstPerfectNumber() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
