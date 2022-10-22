// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class A {
  var var_a: i32;
}

base class B extends A {
  var var_b: i32;
}

fn Main() -> i32 {
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_incorrect_qualification.carbon:[[@LINE+1]]: type error in name binding: '{.var_a: i32, .A.var_b: i32}' is not implicitly convertible to 'class B'
  var b: B = {.var_a=1, .(A.var_b)=2};
  return 0;
}
