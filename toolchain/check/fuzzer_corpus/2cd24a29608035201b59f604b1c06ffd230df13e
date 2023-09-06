// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 1

package ExplorerTest api;
class A(T:! type) {
  var v: T;
}
fn F(T:! type, x: T) -> T {
  var v: A(T) = {.v = x};
  return v.v;
}
fn Main() -> i32 {
  return F(i32, 1);
}
