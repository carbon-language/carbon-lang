// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  var i: i32;
}

fn Foo(c: C) {
  Print("{0}", c.i);
}

fn Main() -> i32 {
  var c: C = { .i = 1 };
  let pinned_c: C = c;
  c.i = 2;
  // CHECK:STDERR: RUNTIME ERROR: fail_pinned_value_subobject_changed.carbon:[[@LINE+1]]: Reference has changed since this value was bound.
  Foo(pinned_c);
  return 0;
}
