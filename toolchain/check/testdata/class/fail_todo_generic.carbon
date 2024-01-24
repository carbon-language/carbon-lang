// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+6]]:1: ERROR: Semantics TODO: `generic class`.
// CHECK:STDERR: class C[]();
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `generic class`.
// CHECK:STDERR: class C[]();
// CHECK:STDERR: ^~~~~~~~~~~~
class C[]();

// CHECK:STDOUT: --- fail_todo_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.C = %C.decl} [template]
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
