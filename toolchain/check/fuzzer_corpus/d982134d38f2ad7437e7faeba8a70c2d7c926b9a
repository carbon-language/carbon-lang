// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 1
// CHECK:STDOUT: 3
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface GetX {
  fn DoIt[self: Self]() -> i32;
}

impl forall [template T:! type] T as GetX {
  fn DoIt[self: Self]() -> i32 { return self.x; }
}

class C {
  var x: i32;
}

fn Main() -> i32 {
  var a: auto = {.x = 1, .y = 2};
  var b: C = {.x = 3};
  Print("{0}", a.(GetX.DoIt)());
  Print("{0}", b.(GetX.DoIt)());
  return 0;
}
