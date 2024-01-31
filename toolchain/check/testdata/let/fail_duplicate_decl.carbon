// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  let a: i32 = 1;
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   let a: i32 = 2;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: Name is previously declared here.
  // CHECK:STDERR:   let a: i32 = 1;
  // CHECK:STDERR:       ^
  let a: i32 = 2;
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %a.loc8: i32 = bind_name a, %.loc8
// CHECK:STDOUT:   %.loc15: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %a.loc15: i32 = bind_name a, %.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
