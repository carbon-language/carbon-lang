// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class C {
  fn DoPrint() { Print("test"); }
}

fn Main() -> i32 {
  var p: auto = heap.New(C);
  var y: auto = *p;
  // The following line would fail with: "could not find `y: auto`"
  // y.DoPrint();
  heap.Delete(p);
  return 0;
}
