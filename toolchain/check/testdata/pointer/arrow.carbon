// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  fn Member[self: Self]();
  var field: C*;
}

fn Foo(ptr: C*) {
  (*ptr).Member();
  ptr->Member();

  (*ptr).field;
  ptr->field;

  ptr->field->field;
}
