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
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+7]]:5: error: semantics TODO: `HandleTemplate` [SemanticsTodo]
// CHECK:STDERR: let template z:! type = i32;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:14: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
// CHECK:STDERR: let template z:! type = i32;
// CHECK:STDERR:              ^~~~~~~~
let template z:! type = i32;
fn FirstPerfectNumber() -> z { return 6; }

// CHECK:STDOUT: --- fail_let_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Six.type: type = fn_type @Six [template]
// CHECK:STDOUT:   %Six: %Six.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 6 [template]
// CHECK:STDOUT:   %HalfDozen.type: type = fn_type @HalfDozen [template]
// CHECK:STDOUT:   %HalfDozen: %HalfDozen.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16: i32 = int_value 6 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HalfDozen() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28: i32 = int_value 6 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
