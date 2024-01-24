// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // CHECK:STDERR: fail_unimplemented_op.carbon:[[@LINE+3]]:10: ERROR: Semantics TODO: `HandleInfixOperatorPlus`.
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
