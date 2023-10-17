// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() {
  // CHECK:STDERR: fail_member_of_function.carbon:[[@LINE+3]]:3: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   A.y;
  // CHECK:STDERR:   ^
  A.y;
}

// CHECK:STDOUT: file "fail_member_of_function.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_reference "A", package.%A
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
