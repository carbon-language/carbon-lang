// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Main() -> (i32, i32) {
  return (15, 35);
}

// CHECK:STDOUT: file "tuple.carbon" {
// CHECK:STDOUT:   %.loc8 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_11.1: i32 = int_literal 15
// CHECK:STDOUT:   %.loc9_11.2: i32 = stub_reference %.loc9_11.1
// CHECK:STDOUT:   %.loc9_15.1: i32 = int_literal 35
// CHECK:STDOUT:   %.loc9_15.2: i32 = stub_reference %.loc9_15.1
// CHECK:STDOUT:   %.loc9_17: (i32, i32) = tuple_literal (%.loc9_11.2, %.loc9_15.2)
// CHECK:STDOUT:   return %.loc9_17
// CHECK:STDOUT: }
