// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+6]]:1: ERROR: Semantics TODO: `generic interface`.
// CHECK:STDERR: interface I[]();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `generic interface`.
// CHECK:STDERR: interface I[]();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
interface I[]();

// CHECK:STDOUT: --- fail_todo_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.I = %I.decl} [template]
// CHECK:STDOUT:   %I.decl = interface_decl @I, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
