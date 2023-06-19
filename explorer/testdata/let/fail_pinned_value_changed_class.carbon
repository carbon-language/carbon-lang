// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  var i: i32;
}

fn Main() -> i32 {
  var a: C = { .i = 1 };
  let b: C = a;
  a.i = 2;
  // CHECK:STDERR: RUNTIME ERROR: fail_pinned_value_changed_class.carbon:[[@LINE+1]]: Value has changed since it was pinned
  Print("{0}", b.i);
  return 0;
}
