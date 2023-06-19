// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Foo(b: i32) {
  Print("{0}", b);
}

fn Main() -> i32 {
  var a: i32 = 1;
  let b: i32 = a;
  a = 2;
  // CHECK:STDERR: RUNTIME ERROR: fail_pinned_value_changed_binding.carbon:[[@LINE+1]]: Value has changed since it was pinned
  Foo(b);
  return 0;
}
