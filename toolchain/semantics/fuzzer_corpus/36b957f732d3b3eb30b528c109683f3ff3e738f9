// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var a: (i32, i32) = (1, 2);
  var p: (i32, i32)* = &a;

  a[0] = 3;
  Print("{0}", (*p)[0]);

  (*p)[1] = 4;
  Print("{0}", a[1]);

  return 0;
}
