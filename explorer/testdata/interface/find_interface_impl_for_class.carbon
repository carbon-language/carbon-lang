// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: (
// CHECK:STDOUT: test
// CHECK:STDOUT: )
// CHECK:STDOUT: result: 42

package ExplorerTest api;

interface Printable {
  fn PrintIt[self: Self]();
}

impl String as Printable {
  fn PrintIt[self: String]() {
    Print(self);
  }
}

class Vector(T:! type) {
  var x: T;
}

// Conditionally implement the API for certain `T`s.
impl forall [U:! Printable] Vector(U) as Printable {
  fn PrintIt[self: Self]() {
    Print("(");
    self.x.PrintIt();
    Print(")");
  }
}

fn Main() -> i32 {
  var v: Vector(String) = {.x = "test"};
  v.PrintIt();
  return 42;
}
