// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: var_a=1
// CHECK:STDOUT: var_b=2
// CHECK:STDOUT: var_a=3
// CHECK:STDOUT: var_b=4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class A {
  var var_a: i32;
}

base class B extends A {
  var var_b: i32;
}

fn Main() -> i32 {
  var b: B = {.(A.var_a)=1, .(B.var_b)=2};
  Print("var_a={0}", b.var_a);
  Print("var_b={0}", b.var_b);

  b = {.(A.var_a)=3, .(B.var_b)=4};
  Print("var_a={0}", b.var_a);
  Print("var_b={0}", b.var_b);
  return 0;
}
