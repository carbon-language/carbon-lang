// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Mismatch() -> i32 {
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE+3]]:19: ERROR: Type `f64` of `returned var` does not match return type `i32` of enclosing function.
  // CHECK:STDERR:   returned var v: f64 = 0.0;
  // CHECK:STDERR:                   ^
  returned var v: f64 = 0.0;
  return 0;
}

// CHECK:STDOUT: file "fail_returned_var_type.carbon" {
// CHECK:STDOUT:   %Mismatch: <function> = fn_decl @Mismatch
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mismatch() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v: f64 = bind_name "v", <error>
// CHECK:STDOUT:   %.loc11: f64 = real_literal 0e-1
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   %.loc12: i32 = int_literal 0
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
