// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Should work with compile-time evaluation.
// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+9]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+6]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+3]]:22: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:                      ^~~~~~
let x: i32 = if true then 1 else 0;

class C {
  // TODO: Should work with compile-time evaluation.
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+12]]:10: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+9]]:10: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+6]]:10: ERROR: Cannot evaluate type expression.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+3]]:18: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:                  ^~~~~~~~
  var n: if true then i32 else f64;
}

// CHECK:STDOUT: --- fail_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc17: i32 = block_arg <unexpected instblockref block5>
// CHECK:STDOUT:   %x: i32 = bind_name x, %.loc17
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc33: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc33 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .n = <unexpected instref inst+21>
// CHECK:STDOUT: }
// CHECK:STDOUT:
