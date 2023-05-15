// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: overflow:1: possible infinite loop: too many interpreter steps executed

package EmptyIdentifier impl;

fn Main() -> i32 {
  while (true) {
    // Ideally we would hit an OOM here, but it's too difficult to OOM from heap
    // allocations before hitting max steps. Maybe with string operations we
    // could trigger actual excessive memory allocations by just doubling the
    // size of the string each time.
    heap.New((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0));
  }
  return 0;
}
