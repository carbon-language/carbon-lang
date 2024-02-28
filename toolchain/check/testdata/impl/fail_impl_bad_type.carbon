// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {}

// CHECK:STDERR: fail_impl_bad_type.carbon:[[@LINE+3]]:6: ERROR: Cannot implicitly convert from `bool` to `type`.
// CHECK:STDERR: impl true as I {}
// CHECK:STDERR:      ^~~~
impl true as I {}

// CHECK:STDOUT: --- fail_impl_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.I = %I.decl} [template]
// CHECK:STDOUT:   %I.decl = interface_decl @I, () [template = constants.%.1]
// CHECK:STDOUT:   impl_decl @impl, (<unexpected instref inst+3>, <unexpected instref inst+5>)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: <error> as I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
