// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/print/fail_index_text.carbon:[[@LINE+1]]: Index invalid with argument count of 0 at offset 8 in `Print: {1a}`
  Print("Print: {1a}");
  return 0;
}
