// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+7]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: i32;
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var A: i32;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+6]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: i32 = 1;
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-14]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
var A: i32 = 1;

// CHECK:STDOUT: --- fail_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc7: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.var.loc16: ref i32 = var A
// CHECK:STDOUT:   %A.loc16: ref i32 = bind_name A, %A.var.loc16
// CHECK:STDOUT:   %A.var.loc24: ref i32 = var A
// CHECK:STDOUT:   %A.loc24: ref i32 = bind_name A, %A.var.loc24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%A.var.loc24, %.loc24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
