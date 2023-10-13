// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: i32) {
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   let a: i32 = 1;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:6: Name is previously declared here.
  // CHECK:STDERR: fn F(a: i32) {
  // CHECK:STDERR:      ^
  let a: i32 = 1;
}

// CHECK:STDOUT: file "fail_duplicate_decl.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.loc7: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 1
// CHECK:STDOUT:   %a.loc14: i32 = bind_name "a", %.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
