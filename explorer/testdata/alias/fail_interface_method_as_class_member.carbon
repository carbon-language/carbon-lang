// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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

  alias PrintIt = Printable.PrintIt;
}

// Conditionally implement the API for certain `T`s.
impl forall [U:! Printable] Vector(U) as Printable {
  fn PrintIt[self: Self]() {
    Print("{ ");
    self.x.PrintIt();
    Print(" }");
  }
}

fn Main() -> i32 {
  var v: Vector(String) = {.x = "test"};
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/alias/fail_interface_method_as_class_member.carbon:[[@LINE+1]]: Member access to aliases is not yet supported.
  v.PrintIt();
  return 42;
}
