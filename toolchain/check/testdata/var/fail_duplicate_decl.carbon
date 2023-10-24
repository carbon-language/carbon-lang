// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Main() {
  var x: i32 = 0;
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: Name is previously declared here.
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^
  var x: i32 = 0;
}

// CHECK:STDOUT: file "fail_duplicate_decl.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.loc9_7.1: ref i32 = var "x"
// CHECK:STDOUT:   %x.loc9_7.2: ref i32 = bind_name "x", %x.loc9_7.1
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.loc9_7.1, %.loc9
// CHECK:STDOUT:   %x.loc16_7.1: ref i32 = var "x"
// CHECK:STDOUT:   %x.loc16_7.2: ref i32 = bind_name "x", %x.loc16_7.1
// CHECK:STDOUT:   %.loc16: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.loc16_7.1, %.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
