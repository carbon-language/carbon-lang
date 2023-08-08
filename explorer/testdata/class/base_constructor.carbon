// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

base class A {
  fn Create() -> Self {
    return {.value_a = 1};
  }
  var value_a: i32;
}

class B {
  extend base: A;
  fn Create() -> Self {
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

// CHECK:STDOUT: 1
// CHECK:STDOUT: 2
// CHECK:STDOUT: result: 0
