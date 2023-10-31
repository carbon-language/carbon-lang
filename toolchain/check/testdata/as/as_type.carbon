// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let t: type = (i32, i32) as type;

// CHECK:STDOUT: file "as_type.carbon" {
// CHECK:STDOUT:   %.loc7_24.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_24.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_26: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %t: type = bind_name "t", %.loc7_26
// CHECK:STDOUT: }
