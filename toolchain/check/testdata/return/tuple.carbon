// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "tuple.carbon" {


fn Main() -> (i32, i32) {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
  return (15, 35);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: i32 = int_literal 15
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: i32 = stub_reference %.loc[[@LINE-2]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_15.1: i32 = int_literal 35
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_15.2: i32 = stub_reference %.loc[[@LINE-4]]_15.1
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_17: (i32, i32) = tuple_literal (%.loc[[@LINE-5]]_11.2, %.loc[[@LINE-5]]_15.2)
  // CHECK:STDOUT:   return %.loc[[@LINE-6]]_17
  // CHECK:STDOUT: }
}
