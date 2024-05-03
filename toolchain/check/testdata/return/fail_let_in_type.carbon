// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let x: type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:13: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn Six() -> x { return 6; }

// TODO: This should probably work.
let y:! type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+4]]:23: ERROR: Cannot implicitly convert from `i32` to `y`.
// CHECK:STDERR: fn HalfDozen() -> y { return 6; }
// CHECK:STDERR:                       ^~~~~~~~~
// CHECK:STDERR:
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
// CHECK:STDOUT:   %Six: type = fn_type @Six [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Six = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %y: type = bind_symbolic_name y 0 [symbolic]
// CHECK:STDOUT:   %HalfDozen: type = fn_type @HalfDozen [template]
// CHECK:STDOUT:   %struct.2: HalfDozen = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HalfDozen() -> y {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20: i32 = int_literal 6 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
