// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {}

// CHECK:STDERR: fail_extend_impl_scope.carbon:[[@LINE+3]]:1: ERROR: `extend impl` can only be used in a class.
// CHECK:STDERR: extend impl i32 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
extend impl i32 as I {}

// CHECK:STDOUT: --- fail_extend_impl_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.I = %I.decl} [template]
// CHECK:STDOUT:   %I.decl = interface_decl @I, () [template = constants.%.1]
// CHECK:STDOUT:   impl_decl @impl, (<unexpected instref inst+3>)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
