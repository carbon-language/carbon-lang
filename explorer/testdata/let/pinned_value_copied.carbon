// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var a: i32 = 1;
  let a_pinned: i32 = a;
  let a_pinned_copy: i32 = a_pinned;

  // OK: Value unused after being mutated.
  a = 2;

  Print("a: {0}", a);

  // OK: Value was copied from `a_pinned`, and reflects previous value.
  // TODO: Avoid value->value copy when possible.
  Print("a_pinned2: {0}", a_pinned_copy);

  return 0;
}

// CHECK:STDOUT: a: 2
// CHECK:STDOUT: a_pinned2: 1
// CHECK:STDOUT: result: 0
