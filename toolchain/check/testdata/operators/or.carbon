// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "or.carbon" {

fn F() -> bool { return true; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
fn G() -> bool { return true; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G

fn Or() -> bool {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Or
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-11]]: bool = bool_literal true
// CHECK:STDOUT:   return %.loc[[@LINE-12]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-15]]: bool = bool_literal true
// CHECK:STDOUT:   return %.loc[[@LINE-16]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> bool {
// CHECK:STDOUT: !entry:
  return F() or G();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: init bool = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: ref bool = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_11.3: ref bool = temporary %.loc[[@LINE-3]]_11.2, %.loc[[@LINE-3]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.4: bool = bind_value %.loc[[@LINE-4]]_11.3
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_14.1: bool = not %.loc[[@LINE-5]]_11.4
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_14.2: bool = bool_literal true
  // CHECK:STDOUT:   if %.loc[[@LINE-7]]_14.1 br !or.rhs else br !or.result(%.loc[[@LINE-7]]_14.2)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !or.rhs:
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_18.1: init bool = call @G()
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_18.2: ref bool = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-12]]_18.3: ref bool = temporary %.loc[[@LINE-12]]_18.2, %.loc[[@LINE-12]]_18.1
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_18.4: bool = bind_value %.loc[[@LINE-13]]_18.3
  // CHECK:STDOUT:   br !or.result(%.loc[[@LINE-14]]_18.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !or.result:
  // CHECK:STDOUT:   %.loc[[@LINE-17]]_14.3: bool = block_arg !or.result
  // CHECK:STDOUT:   return %.loc[[@LINE-18]]_14.3
  // CHECK:STDOUT: }
}
