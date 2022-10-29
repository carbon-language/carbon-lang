// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class A {
  fn Create() -> Self {
    return {.value_a = 1};
  }
  var value_a: i32;
}

class B extends A {
  fn Create() -> Self {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_base_constructor.carbon:[[@LINE+1]]: error converting struct to class: using the result of a function call for the `.base` field is not supported yet.
    return {.base = A.Create(), .value_b = 2};
  }
  var value_b: i32;
}

fn Main() -> i32 {
  var b: B = B.Create();
  Print("{0}", b.value_a);
  Print("{0}", b.value_b);
  return 0;
}
