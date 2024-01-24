// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private let a: i32 = 1;
// CHECK:STDERR: ^~~~~~~
private let a: i32 = 1;

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %.loc10: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
