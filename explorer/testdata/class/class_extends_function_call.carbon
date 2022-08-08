// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Func1
// CHECK: Func2
// CHECK: result: 0

package ExplorerTest api;

class C {
  fn Foo(v: i32) {
    Print("Func{0}", v);
  }
  fn Func1[me:Self]() {
    me.Foo(1);
  }
}

class D extends C {
  fn Func2[me:Self]() {
    me.Foo(2);
  }
}

fn Main() -> i32 {
  var d2: D = {};
  d2.Func1();
  d2.Func2();
  return 0;
}
