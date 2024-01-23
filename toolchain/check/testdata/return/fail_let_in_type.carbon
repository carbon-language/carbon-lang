// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let x: type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:13: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
fn Six() -> x { return 6; }

// TODO: This should probably work.
let y:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:23: ERROR: Cannot implicitly convert from `i32` to `y`.
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
fn HalfDozen() -> y { return 6; }

// TODO: This should work.
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:5: ERROR: Semantics TODO: `HandleTemplate`.
// CHECK:STDERR: let template z:! type = i32;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
let template z:! type = i32;
fn FirstPerfectNumber() -> z { return 6; }

// CHECK:STDOUT: --- fail_let_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 6 [template = constants.%.1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HalfDozen() -> y {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 6 [template = constants.%.1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
