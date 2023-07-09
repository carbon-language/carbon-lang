// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  let original: i32 = 1;
  var a: i32 = original;
  let a_pinned: i32 = a;
  a = 2;
  a = original;
  // CHECK:STDERR: RUNTIME ERROR: fail_pinned_value_restored.carbon:[[@LINE+1]]: Value has changed since it was pinned
  Print("{0}", a_pinned);
  return 0;
}
