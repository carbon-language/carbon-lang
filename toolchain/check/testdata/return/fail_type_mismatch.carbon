// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:13: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   return 1.0;
  // CHECK:STDERR:             ^
  return 1.0;
}

// CHECK:STDOUT: file "fail_type_mismatch.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: f64 = real_literal 10e-1
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
