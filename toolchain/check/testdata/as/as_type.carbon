// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let t: type = (i32, i32) as type;

// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {}
// CHECK:STDOUT:   %.loc7: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.2: type = converted %.loc7, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %t: type = bind_name t, %.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
