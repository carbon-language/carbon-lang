// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {}

impl i32 as I {}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `impl i32 as I`.
// CHECK:STDERR: impl i32 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-5]]:1: Previous definition was here.
// CHECK:STDERR: impl i32 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
impl i32 as I {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.I = %I.decl} [template]
// CHECK:STDOUT:   %I.decl = interface_decl @I, ()
// CHECK:STDOUT:   impl_decl @impl, (<unexpected instref inst+3>)
// CHECK:STDOUT:   impl_decl @impl, (<unexpected instref inst+5>)
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
