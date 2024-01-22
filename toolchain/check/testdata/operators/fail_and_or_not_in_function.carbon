// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+9]]:11: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:           ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+6]]:13: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+3]]:13: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
var and_: F(true and true);

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+9]]:10: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+6]]:12: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+3]]:12: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
var or_: F(true or true);

// CHECK:STDOUT: --- fail_and_or_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc33_17: bool = block_arg <unexpected instblockref block13>
// CHECK:STDOUT:   %.loc33_11.1: init type = call <unexpected instref inst+27>(%.loc33_17)
// CHECK:STDOUT:   %.loc33_24: type = value_of_initializer %.loc33_11.1
// CHECK:STDOUT:   %.loc33_11.2: type = converted %.loc33_11.1, %.loc33_24
// CHECK:STDOUT:   %or_.var: ref <error> = var or_
// CHECK:STDOUT:   %or_: ref <error> = bind_name or_, %or_.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   br !if.expr.result(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   br !if.expr.result(f64)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc9: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
