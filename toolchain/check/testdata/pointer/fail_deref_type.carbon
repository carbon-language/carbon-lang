// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+6]]:8: ERROR: Cannot dereference operand of non-pointer type `type`.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+3]]:8: To form a pointer type, write the `*` after the pointee type.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
var p: *i32;

// CHECK:STDOUT: --- fail_deref_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.p = %p} [template]
// CHECK:STDOUT:   %.loc13: ref <error> = deref i32
// CHECK:STDOUT:   %p.var: ref <error> = var p
// CHECK:STDOUT:   %p: ref <error> = bind_name p, %p.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
