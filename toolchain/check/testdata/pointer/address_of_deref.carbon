// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "address_of_deref.carbon" {

fn F() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n: ref i32 = var "n"
  var n: i32 = 0;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 0
  // CHECK:STDOUT:   assign %n, %.loc[[@LINE-2]]
  return *&*&n;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_13.1: type = ptr_type i32
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_13.2: i32* = address_of %n
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_12: ref i32 = dereference %.loc[[@LINE-3]]_13.2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11: i32* = address_of %.loc[[@LINE-4]]_12
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_10.1: ref i32 = dereference %.loc[[@LINE-5]]_11
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_10.2: i32 = bind_value %.loc[[@LINE-6]]_10.1
  // CHECK:STDOUT:   return %.loc[[@LINE-7]]_10.2
  // CHECK:STDOUT: }
}
