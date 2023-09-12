// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_duplicate_decl.carbon" {


fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  var x: i32 = 0;
  // CHECK:STDOUT:   %x.loc[[@LINE-1]]: ref i32 = var "x"
  // CHECK:STDOUT:   %.loc[[@LINE-2]]: i32 = int_literal 0
  // CHECK:STDOUT:   assign %x.loc[[@LINE-3]], %.loc[[@LINE-3]]
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-7]]:7: Name is previously declared here.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  var x: i32 = 0;
  // CHECK:STDOUT:   %x.loc[[@LINE-1]]: ref i32 = var "x"
  // CHECK:STDOUT:   %.loc[[@LINE-2]]: i32 = int_literal 0
  // CHECK:STDOUT:   assign %x.loc[[@LINE-3]], %.loc[[@LINE-3]]
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
