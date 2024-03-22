// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

alias a = i32;
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var a: i32 = 1;
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:7: Name is previously declared here.
// CHECK:STDERR: alias a = i32;
// CHECK:STDERR:       ^
// CHECK:STDERR:
var a: i32 = 1;

var b: i32 = 2;
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: alias b = i32;
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:5: Name is previously declared here.
// CHECK:STDERR: var b: i32 = 2;
// CHECK:STDERR:     ^
alias b = i32;

// CHECK:STDOUT: --- fail_name_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a.loc7
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.loc7: type = bind_alias a, i32 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a.loc15: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc24: type = bind_alias <invalid>, i32 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc15
// CHECK:STDOUT:   %.loc17: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
