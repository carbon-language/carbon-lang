// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // The following line has two mismatches, but after the first, it shouldn't
  // keep erroring.
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+3]]:13: ERROR: Cannot implicitly convert from `i32` to `f64`.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:             ^
  return 12 + 3.4 + 12;
}

// CHECK:STDOUT: file "fail_type_mismatch_once.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_10: i32 = int_literal 12
// CHECK:STDOUT:   %.loc13_15: f64 = real_literal 34e-1
// CHECK:STDOUT:   %.loc13_13: <error> = add <error>, %.loc13_15
// CHECK:STDOUT:   %.loc13_21: i32 = int_literal 12
// CHECK:STDOUT:   %.loc13_19: <error> = add <error>, %.loc13_21
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
