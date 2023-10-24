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
// CHECK:STDOUT:   %x.loc10_5.1: ref type = var "x"
// CHECK:STDOUT:   %x.loc10_5.2: ref type = bind_name "x", %x.loc10_5.1
// CHECK:STDOUT:   %.loc10: i32 = int_literal 42
// CHECK:STDOUT:   assign %x.loc10_5.1, <error>
// CHECK:STDOUT: }
