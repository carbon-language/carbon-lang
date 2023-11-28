// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Procedure() -> i32 {
  // CHECK:STDERR: fail_return_var_no_returned_var.carbon:[[@LINE+3]]:3: ERROR: `return var;` with no `returned var` in scope.
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:   ^~~~~~~~~~~
  return var;
}

// CHECK:STDOUT: file "fail_return_var_no_returned_var.carbon" {
// CHECK:STDOUT:   %Procedure: <function> = fn_decl @Procedure
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
