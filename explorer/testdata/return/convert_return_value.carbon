// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  extend impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return 42; }
  }
}

fn Main() -> i32 {
  var a: A = {};
  return a;
}

// CHECK:STDOUT: result: 42
