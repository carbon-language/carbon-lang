// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_dereference_type.carbon:[[@LINE+6]]:8: ERROR: Cannot dereference operand of non-pointer type `type`.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_dereference_type.carbon:[[@LINE+3]]:8: To form a pointer type, write the `*` after the pointee type.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
var p: *i32;

// CHECK:STDOUT: file "fail_dereference_type.carbon" {
// CHECK:STDOUT:   %.loc13: ref <error> = dereference i32
// CHECK:STDOUT:   %p: ref <error> = var "p"
// CHECK:STDOUT: }
