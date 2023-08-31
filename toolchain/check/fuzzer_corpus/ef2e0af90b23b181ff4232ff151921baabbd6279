// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: test
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn F() { Print("test"); }

fn Main() -> i32 {
  var p: auto = heap.New(F);
  var y: auto = *p;
  y();
  heap.Delete(p);
  return 0;
}
