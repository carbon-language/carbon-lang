// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+3]]:17: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR:                 ^
var x: type = 42;

// CHECK:STDOUT: file "fail_non_type_as_type.carbon" {
// CHECK:STDOUT:   %x: ref type = var "x"
// CHECK:STDOUT:   %.loc10: i32 = int_literal 42
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
