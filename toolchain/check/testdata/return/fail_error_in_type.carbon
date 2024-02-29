// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_error_in_type.carbon:[[@LINE+3]]:13: ERROR: Name `x` not found.
// CHECK:STDERR: fn Six() -> x;
// CHECK:STDERR:             ^
fn Six() -> x;

// CHECK:STDOUT: --- fail_error_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Six = %Six
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Six: <function> = fn_decl @Six [template] {
// CHECK:STDOUT:     %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error>;
// CHECK:STDOUT:
