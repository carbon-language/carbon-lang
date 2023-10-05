// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+3]]:24: ERROR: Cannot implicitly convert from `type` to `i32`.
// CHECK:STDERR: var x: (i32, ) = (i32, );
// CHECK:STDERR:                        ^
var x: (i32, ) = (i32, );

// CHECK:STDOUT: file "fail_type_assign.carbon" {
// CHECK:STDOUT:   %.loc10_14.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc10_14.2: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc10_14.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: ref (i32,) = var "x"
// CHECK:STDOUT:   %.loc10_24: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
