// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Foo(n: i32) {
  Print("{0}", n);
}

fn Main() -> i32 {
  var v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  let a: i32 = v.a;
  v.b = 3;
  Foo(a);
  return 0;
}
