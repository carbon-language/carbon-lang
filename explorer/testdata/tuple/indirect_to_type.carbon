// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  impl as ImplicitAs(type) {
    fn Convert[self: Self]() -> type { return i32; }
  }
}

fn MakePair() -> (A, A) { return ({}, {}); }

fn Main() -> i32 {
  let b: MakePair() = (1, 2);
  Print("1: {0}", b[0]);
  Print("2: {0}", b[1]);
  return 0;
}

// CHECK:STDOUT: 1: 1
// CHECK:STDOUT: 2: 2
// CHECK:STDOUT: result: 0
