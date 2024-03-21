// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_impl_bad_interface.carbon:[[@LINE+7]]:1: ERROR: Semantics TODO: `impl as non-interface`.
// CHECK:STDERR: impl i32 as false {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_impl_bad_interface.carbon:[[@LINE+3]]:13: ERROR: Cannot implicitly convert from `bool` to `type`.
// CHECK:STDERR: impl i32 as false {}
// CHECK:STDERR:             ^~~~~
impl i32 as false {}

// CHECK:STDOUT: --- fail_impl_bad_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %.loc14: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
