// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  fn F[self: Self]() -> i32;
}

fn Main() -> i32 {
  var a: A = {};
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/class/fail_call_undefined_method.carbon:[[@LINE+1]]: attempt to call function `F` that has not been defined
  return a.F();
}
