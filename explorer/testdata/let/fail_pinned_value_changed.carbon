// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var a: i32 = 1;
  let a_pinned: i32 = a;
  a = 2;
  // CHECK:STDERR: RUNTIME ERROR: fail_pinned_value_changed.carbon:[[@LINE+1]]: Reference has changed since this value was bound.
  Print("{0}", a_pinned);
  return 0;
}
