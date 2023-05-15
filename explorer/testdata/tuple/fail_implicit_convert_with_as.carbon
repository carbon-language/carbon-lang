// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  var a: i32;
}
class B {
  var b: i32;
}

external impl A as As(B) {
  fn Convert[self: Self]() -> B { return {.b = self.a}; }
}

fn Main() -> i32 {
  var a: (i32, A) = (1, {.a = 2});
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/tuple/fail_implicit_convert_with_as.carbon:[[@LINE+1]]: type error in initializer of variable: '(i32, class A)' is not implicitly convertible to '(i32, class B)'
  var b: (i32, B) = a;
  return b[1].b;
}
