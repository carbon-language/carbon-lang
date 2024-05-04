// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn While() {
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+3]]:9: ERROR: Cannot implicitly convert from `String` to `bool`.
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  while ("Hello") {}
}

// CHECK:STDOUT: --- fail_bad_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While: type = fn_type @While [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: While = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type String [template]
// CHECK:STDOUT:   %.3: String = string_literal "Hello" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %While.decl: While = fn_decl @While [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %.loc11: String = string_literal "Hello" [template = constants.%.3]
// CHECK:STDOUT:   if <error> br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
