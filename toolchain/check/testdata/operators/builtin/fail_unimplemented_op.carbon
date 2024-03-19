// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // CHECK:STDERR: fail_unimplemented_op.carbon:[[@LINE+3]]:10: ERROR: Semantics TODO: `missing or invalid operator interface`.
  // CHECK:STDERR:   return 12 + 34;
  // CHECK:STDERR:          ^~~~~~~
  return 12 + 34;
}

// CHECK:STDOUT: --- fail_unimplemented_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 34 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Main = %Main
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template] {
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_10: i32 = int_literal 12 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_15: i32 = int_literal 34 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
