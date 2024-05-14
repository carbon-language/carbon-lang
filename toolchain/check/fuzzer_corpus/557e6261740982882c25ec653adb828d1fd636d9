// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface ConvertsFromInt {
  require i32 impls ImplicitAs(Self);
}

fn ConvertIntTo(T:! ConvertsFromInt, n: i32) -> T {
  return n;
}

class IntHolder {
  var n: i32;
}
impl i32 as ImplicitAs(IntHolder) {
  fn Convert[self: Self]() -> IntHolder { return {.n = self}; }
}
impl IntHolder as ConvertsFromInt {}

fn Main() -> i32 {
  return ConvertIntTo(IntHolder, 5).n;
}
