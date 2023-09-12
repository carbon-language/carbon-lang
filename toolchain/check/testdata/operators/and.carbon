// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "and.carbon" {

fn F() -> bool { return true; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
fn G() -> bool { return true; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G

fn And() -> bool {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @And
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
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
  return F() and G();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: init bool = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: ref bool = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_11.3: ref bool = temporary %.loc[[@LINE-3]]_11.2, %.loc[[@LINE-3]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.4: bool = bind_value %.loc[[@LINE-4]]_11.3
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_14.1: bool = bool_literal false
  // CHECK:STDOUT:   if %.loc[[@LINE-6]]_11.4 br !and.rhs else br !and.result(%.loc[[@LINE-6]]_14.1)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !and.rhs:
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_19.1: init bool = call @G()
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_19.2: ref bool = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_19.3: ref bool = temporary %.loc[[@LINE-11]]_19.2, %.loc[[@LINE-11]]_19.1
  // CHECK:STDOUT:   %.loc[[@LINE-12]]_19.4: bool = bind_value %.loc[[@LINE-12]]_19.3
  // CHECK:STDOUT:   br !and.result(%.loc[[@LINE-13]]_19.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !and.result:
  // CHECK:STDOUT:   %.loc[[@LINE-16]]_14.2: bool = block_arg !and.result
  // CHECK:STDOUT:   return %.loc[[@LINE-17]]_14.2
  // CHECK:STDOUT: }
}
