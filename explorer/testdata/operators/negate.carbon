// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A { var n: i32; }

impl A as Negate where .Result = A {
  fn Op[self: Self]() -> A { return {.n = -self.n}; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  a = -a;
  return a.n;
}

// CHECK:STDOUT: result: -5
